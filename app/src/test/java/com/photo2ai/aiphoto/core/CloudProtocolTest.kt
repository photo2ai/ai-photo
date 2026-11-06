package com.photo2ai.aiphoto.core

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.util.Base64

class CloudProtocolTest {
    private val png = byteArrayOf(-119, 80, 78, 71, 1, 2, 3)
    private val b64 = Base64.getEncoder().encodeToString(png)

    private fun failure(block: () -> Unit): String {
        try {
            block()
        } catch (e: CloudProtocol.Failure) {
            return e.message!!
        }
        fail("expected Failure")
        error("unreachable")
    }

    @Test
    fun geminiUrlAddsVersionOnlyWhenMissing() {
        assertEquals(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent",
            CloudProtocol.geminiUrl("https://generativelanguage.googleapis.com/", "gemini-2.5-flash-image"),
        )
        assertEquals("https://proxy.example/v1beta/models/m:generateContent", CloudProtocol.geminiUrl("https://proxy.example/v1beta", "m"))
        assertEquals("http://10.0.2.2:8080/v1/models/m:generateContent", CloudProtocol.geminiUrl("http://10.0.2.2:8080/v1/", " m "))
    }

    @Test
    fun geminiBodyHasImageAndText() {
        val o = JSONObject(CloudProtocol.geminiBody("hi", png, "image/jpeg"))
        val parts = o.getJSONArray("contents").getJSONObject(0).getJSONArray("parts")
        assertEquals(b64, parts.getJSONObject(0).getJSONObject("inline_data").getString("data"))
        assertEquals("hi", parts.getJSONObject(1).getString("text"))
        assertEquals("IMAGE", o.getJSONObject("generationConfig").getJSONArray("responseModalities").getString(1))
    }

    @Test
    fun geminiBodyKeepsImageOrderForTwoImages() {
        val second = byteArrayOf(9, 8, 7)
        val parts = JSONObject(CloudProtocol.geminiBody("p", listOf(png to "image/jpeg", second to "image/png")))
            .getJSONArray("contents").getJSONObject(0).getJSONArray("parts")
        assertEquals(3, parts.length())
        assertEquals(b64, parts.getJSONObject(0).getJSONObject("inline_data").getString("data"))
        assertEquals("image/png", parts.getJSONObject(1).getJSONObject("inline_data").getString("mime_type"))
        assertEquals("p", parts.getJSONObject(2).getString("text"))
    }

    @Test
    fun outfitPromptWithAndWithoutReference() {
        val ref = CloudProtocol.outfitPrompt("", hasRef = true)
        assertTrue(ref.contains("Image 2") && !ref.contains("()"))
        assertTrue(CloudProtocol.outfitPrompt("红色毛衣", hasRef = true).contains("(红色毛衣)"))
        val text = CloudProtocol.outfitPrompt(" 白色婚纱 ", hasRef = false)
        assertTrue(text.contains("to: 白色婚纱.") && !text.contains("Image 2"))
        assertTrue(CloudProtocol.FACE_SWAP_PROMPT.contains("Image 2"))
    }

    @Test
    fun parseGeminiTakesLastNonThoughtImage() {
        val json = """{"candidates":[{"content":{"parts":[
            {"text":"here you go"},
            {"inlineData":{"mimeType":"image/png","data":"AAAA"},"thought":true},
            {"inlineData":{"mimeType":"image/png","data":"$b64"}}]},"finishReason":"STOP"}]}"""
        assertArrayEquals(png, CloudProtocol.parseGemini(json))
        val snake = """{"candidates":[{"content":{"parts":[{"inline_data":{"mime_type":"image/png","data":"$b64"}}]}}]}"""
        assertArrayEquals(png, CloudProtocol.parseGemini(snake))
    }

    @Test
    fun parseGeminiExplainsFailures() {
        assertTrue(failure { CloudProtocol.parseGemini("""{"error":{"code":400,"message":"API key not valid"}}""") }.contains("API key not valid"))
        assertTrue(failure { CloudProtocol.parseGemini("""{"promptFeedback":{"blockReason":"SAFETY"}}""") }.contains("SAFETY"))
        assertTrue(failure { CloudProtocol.parseGemini("""{"candidates":[{"content":{"parts":[{"text":"I can't edit this"}]},"finishReason":"STOP"}]}""") }.contains("I can't edit this"))
        assertTrue(failure { CloudProtocol.parseGemini("""{"candidates":[{"finishReason":"IMAGE_SAFETY"}]}""") }.contains("安全"))
        assertTrue(failure { CloudProtocol.parseGemini("<html>502</html>") }.contains("不是 JSON"))
    }

    @Test
    fun parseOpenAi() {
        assertArrayEquals(png, CloudProtocol.parseOpenAi("""{"data":[{"b64_json":"$b64"}]}""").first)
        assertEquals("https://x/y.png", CloudProtocol.parseOpenAi("""{"data":[{"url":"https://x/y.png"}]}""").second)
        assertTrue(failure { CloudProtocol.parseOpenAi("""{"error":{"message":"Invalid API key"}}""") }.contains("Invalid API key"))
        assertEquals("https://api.openai.com/v1/images/edits", CloudProtocol.openAiUrl("https://api.openai.com/v1/"))
    }

    @Test
    fun httpErrorUsesServiceMessage() {
        val m = CloudProtocol.httpError(401, """{"error":{"message":"bad key"}}""")
        assertTrue(m, m.contains("API Key") && m.contains("401") && m.contains("bad key"))
        assertEquals("服务暂时出错（HTTP 503）", CloudProtocol.httpError(503, "oops"))
    }
}
