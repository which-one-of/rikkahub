package me.rerere.ai.provider.providers.claude

import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import me.rerere.ai.core.MessageRole
import me.rerere.ai.provider.ClaudePromptCacheTtl
import me.rerere.ai.provider.Model
import me.rerere.ai.provider.TextGenerationResult
import me.rerere.ai.provider.stream.SseEvent
import me.rerere.ai.ui.StreamChunkHandler
import me.rerere.ai.ui.UIMessage
import me.rerere.ai.ui.UIMessagePart
import me.rerere.ai.ui.handleTextGenerationResult
import me.rerere.ai.util.json
import okhttp3.OkHttpClient
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Offline replay-contract tests against production parsing, merging and request building.
 * Signatures are synthetic opaque strings: these tests check preservation, not cryptography
 * or acceptance by a live service. Assertions describe the required replay-preservation behavior.
 */
class ClaudeThinkingReplayTest {
    private val provider = ClaudeProvider(OkHttpClient())

    private fun blocks(value: String): JsonArray = json.parseToJsonElement(value).jsonArray

    private fun thinking(text: String, signature: String): JsonObject =
        blocks("""[{"type":"thinking","thinking":"$text","signature":"$signature"}]""").single().jsonObject

    private fun request(messages: List<UIMessage>): JsonArray {
        val method = ClaudeProvider::class.java.getDeclaredMethod(
            "buildMessages",
            List::class.java,
            Boolean::class.javaPrimitiveType,
            ClaudePromptCacheTtl::class.java,
        )
        method.isAccessible = true
        return method.invoke(provider, messages, false, ClaudePromptCacheTtl.FIVE_MINUTES) as JsonArray
    }

    private fun assistantBlocks(messages: List<UIMessage>): List<JsonObject> =
        request(messages).filter { it.jsonObject["role"]?.jsonPrimitive?.content == "assistant" }
            .flatMap { it.jsonObject.getValue("content").jsonArray.map { block -> block.jsonObject } }

    private fun append(messages: List<UIMessage>, content: JsonArray): List<UIMessage> =
        messages.handleTextGenerationResult(
            TextGenerationResult("synthetic-response", "synthetic-model", provider.parseMessage(content)),
        )

    private fun streamed(signaturePieces: List<String>): List<UIMessage> {
        val decoder = ClaudeStreamDecoder()
        val handler = StreamChunkHandler()
        var messages = listOf(UIMessage.user("first question"))
        fun event(name: String, data: String) {
            decoder.accept(SseEvent(event = name, data = data)).chunks.forEach {
                messages = handler.handle(messages, it)
            }
        }
        event("content_block_start", """{"index":0,"content_block":{"type":"thinking","thinking":""}}""")
        event("content_block_delta", """{"index":0,"delta":{"type":"thinking_delta","thinking":"Thought A"}}""")
        signaturePieces.forEach { piece ->
            event("content_block_delta", """{"index":0,"delta":{"type":"signature_delta","signature":"$piece"}}""")
        }
        event("content_block_stop", """{"index":0}""")
        event("message_stop", "{}")
        return messages + UIMessage.user("second question")
    }

    @Test
    fun `single streaming signature is replayed unchanged`() {
        assertEquals(listOf(thinking("Thought A", "sig-A")), assistantBlocks(streamed(listOf("sig-A"))))
    }

    @Test
    fun `multiple signature deltas preserve the complete signature`() {
        assertEquals(
            listOf(thinking("Thought A", "sig-A")),
            assistantBlocks(streamed(listOf("sig-", "A"))),
        )
    }

    @Test
    fun `nonstream response appended to a user preserves adjacent thinking blocks`() {
        val content = JsonArray(listOf(thinking("Thought A", "sig-A"), thinking("Thought B", "sig-B")))
        val messages = append(listOf(UIMessage.user("first question")), content)
        assertEquals(content.toList(), assistantBlocks(messages + UIMessage.user("second question")))
    }

    @Test
    fun `nonstream provider wrapper preserves adjacent blocks even without pause turn`() = runTest {
        // The provider wrapper, not AgentLoop, merges into an empty assistant on every response.
        val content = JsonArray(listOf(thinking("Thought A", "sig-A"), thinking("Thought B", "sig-B")))
        val history = listOf(UIMessage.user("first question"))
        var calls = 0
        val result = generateClaudeWithPauseTurn(history, Model(modelId = "synthetic-model")) {
            calls++
            TextGenerationResult("response", "synthetic-model", provider.parseMessage(content), "end_turn")
        }
        assertEquals(1, calls)
        assertEquals(content.toList(), assistantBlocks(history + result.message + UIMessage.user("second question")))
    }

    @Test
    fun `pause turn continuation preserves separately signed blocks`() = runTest {
        val history = listOf(UIMessage.user("first question"))
        val first = JsonArray(listOf(thinking("Thought A", "sig-A")))
        val second = JsonArray(listOf(thinking("Thought B", "sig-B")))
        var calls = 0
        val result = generateClaudeWithPauseTurn(history, Model(modelId = "synthetic-model")) { messages ->
            calls++
            if (calls == 1) {
                TextGenerationResult("response-1", "synthetic-model", provider.parseMessage(first), "pause_turn")
            } else {
                assertEquals(first.toList(), assistantBlocks(messages))
                TextGenerationResult("response-2", "synthetic-model", provider.parseMessage(second), "end_turn")
            }
        }
        assertEquals(2, calls)
        assertEquals(
            listOf(thinking("Thought A", "sig-A"), thinking("Thought B", "sig-B")),
            assistantBlocks(history + result.message + UIMessage.user("second question")),
        )
    }

    @Test
    fun `text between signed thinking blocks preserves both blocks`() {
        val content = blocks("""[
            {"type":"thinking","thinking":"Thought A","signature":"sig-A"},
            {"type":"text","text":"separator"},
            {"type":"thinking","thinking":"Thought B","signature":"sig-B"}
        ]""")
        val messages = append(listOf(UIMessage.user("first question"), UIMessage(role = MessageRole.ASSISTANT, parts = emptyList())), content)
        assertEquals(content.toList(), assistantBlocks(messages + UIMessage.user("second question")))
    }

    @Test
    fun `tool boundary preserves thinking blocks and tool result placement`() {
        val firstContent = blocks("""[
            {"type":"thinking","thinking":"Thought A","signature":"sig-A"},
            {"type":"tool_use","id":"call-1","name":"lookup","input":{"q":"test"}}
        ]""")
        val first = append(listOf(UIMessage.user("first question")), firstContent)
        val executed = first.dropLast(1) + first.last().copy(parts = first.last().parts.map {
            if (it is UIMessagePart.Tool) it.copy(output = listOf(UIMessagePart.Text("result"))) else it
        })
        val secondContent = blocks("""[
            {"type":"thinking","thinking":"Thought B","signature":"sig-B"},
            {"type":"text","text":"answer"}
        ]""")
        val messages = append(executed, secondContent) + UIMessage.user("second question")
        assertEquals(firstContent.toList() + secondContent.toList(), assistantBlocks(messages))
        val wire = request(messages)
        assertEquals(listOf("user", "assistant", "user", "assistant", "user"), wire.map {
            it.jsonObject.getValue("role").jsonPrimitive.content
        })
        assertEquals(
            blocks("""[{"type":"tool_result","tool_use_id":"call-1","content":[{"type":"text","text":"result"}]}]"""),
            wire[2].jsonObject.getValue("content"),
        )
    }

    @Test
    fun `redacted thinking survives nonstream parse and replay`() {
        val content = blocks("""[
            {"type":"redacted_thinking","data":"synthetic-opaque-data"},
            {"type":"text","text":"answer"}
        ]""")
        val messages = append(listOf(UIMessage.user("first question")), content)
        assertEquals(content.toList(), assistantBlocks(messages + UIMessage.user("second question")))
    }
    @Test
    fun `metadata only thinking survives storage and request filtering`() {
        val content = blocks("""[
            {"type":"thinking","thinking":"","signature":"opaque-signature"},
            {"type":"redacted_thinking","data":"opaque-data"}
        ]""")
        val parsed = provider.parseMessage(content)
        val restored = json.decodeFromString<UIMessage>(json.encodeToString(UIMessage.serializer(), parsed))
        assertEquals(content.toList(), assistantBlocks(listOf(UIMessage.user("first"), restored, UIMessage.user("second"))))
    }

    @Test
    fun `streamed blocks preserve boundaries signatures initial text and redacted data`() {
        val decoder = ClaudeStreamDecoder()
        val handler = StreamChunkHandler()
        var messages = listOf(UIMessage.user("first"))
        fun event(name: String, data: String) {
            decoder.accept(SseEvent(event = name, data = data)).chunks.forEach {
                messages = handler.handle(messages, it)
            }
        }
        event("content_block_start", """{"index":0,"content_block":{"type":"thinking","thinking":"initial-","signature":"sig-"}}""")
        event("content_block_delta", """{"index":0,"delta":{"type":"thinking_delta","thinking":"A"}}""")
        event("content_block_delta", """{"index":0,"delta":{"type":"signature_delta","signature":"A"}}""")
        event("content_block_stop", """{"index":0}""")
        event("content_block_start", """{"index":1,"content_block":{"type":"thinking","thinking":"B"}}""")
        event("content_block_delta", """{"index":1,"delta":{"type":"signature_delta","signature":"sig-"}}""")
        event("content_block_delta", """{"index":1,"delta":{"type":"signature_delta","signature":"B"}}""")
        event("content_block_stop", """{"index":1}""")
        event("content_block_start", """{"index":2,"content_block":{"type":"redacted_thinking","data":"opaque-data"}}""")
        event("content_block_stop", """{"index":2}""")
        event("message_stop", "{}")
        val restored = json.decodeFromString<UIMessage>(json.encodeToString(UIMessage.serializer(), messages.last()))
        assertEquals(blocks("""[
            {"type":"thinking","thinking":"initial-A","signature":"sig-A"},
            {"type":"thinking","thinking":"B","signature":"sig-B"},
            {"type":"redacted_thinking","data":"opaque-data"}
        ]""").toList(), assistantBlocks(listOf(messages.first(), restored, UIMessage.user("second"))))
    }

}
