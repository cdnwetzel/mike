import type {
    LlmMessage,
    NormalizedToolCall,
    NormalizedToolResult,
    OpenAIToolSchema,
    StreamChatParams,
    StreamChatResult,
} from "./types";

const DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1";
const MAX_OUTPUT_TOKENS = 16384;

type ChatCompletionFunction = {
    name: string;
    arguments: string;
};

type ChatCompletionToolCall = {
    id: string;
    type: "function";
    function: ChatCompletionFunction;
};

type ChatCompletionMessage =
    | {
          role: "system" | "user" | "assistant";
          content: string;
          tool_calls?: ChatCompletionToolCall[];
      }
    | { role: "tool"; tool_call_id: string; content: string };

type ChatCompletionTool = {
    type: "function";
    function: {
        name: string;
        description?: string;
        parameters: Record<string, unknown>;
    };
};

type ChatCompletionChunk = {
    choices?: {
        delta?: {
            content?: string;
            tool_calls?: {
                index: number;
                id?: string;
                function?: {
                    name?: string;
                    arguments?: string;
                };
            }[];
        };
    }[];
};

function apiKey(override?: string | null): string {
    return override?.trim() || process.env.OPENAI_API_KEY?.trim() || "";
}

function openAIBaseUrl(): string {
    const configured =
        process.env.OPENAI_BASE_URL?.trim() ||
        process.env.OPENAI_API_BASE_URL?.trim();
    return (configured || DEFAULT_OPENAI_BASE_URL).replace(/\/+$/, "");
}

function chatCompletionsUrl(): string {
    return `${openAIBaseUrl()}/chat/completions`;
}

function selectedModel(model: string): string {
    return (
        process.env.OPENAI_MODEL_OVERRIDE?.trim() ||
        process.env.OPENAI_MODEL?.trim() ||
        model
    );
}

function toChatCompletionMessages(messages: LlmMessage[]): ChatCompletionMessage[] {
    return messages.map((message) => ({
        role: message.role,
        content: message.content,
    }));
}

function toChatCompletionTools(tools: OpenAIToolSchema[]): ChatCompletionTool[] {
    return tools.map((tool) => ({
        type: "function",
        function: {
            name: tool.function.name,
            description: tool.function.description,
            parameters: tool.function.parameters,
        },
    }));
}

function extractSseJson(buffer: string): { events: unknown[]; rest: string } {
    const events: unknown[] = [];
    const chunks = buffer.split(/\n\n/);
    const rest = chunks.pop() ?? "";

    for (const chunk of chunks) {
        const dataLines = chunk
            .split("\n")
            .map((line) => line.trim())
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trim());

        for (const data of dataLines) {
            if (!data || data === "[DONE]") continue;
            try {
                events.push(JSON.parse(data));
            } catch {
                // Incomplete events stay buffered until the next read.
            }
        }
    }

    return { events, rest };
}

function parseFunctionCall(rawCall: {
    id?: string;
    name?: string;
    arguments?: string;
}): NormalizedToolCall {
    let parsedInput: Record<string, unknown> = {};
    try {
        const parsed = JSON.parse(rawCall.arguments || "{}");
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
            parsedInput = parsed as Record<string, unknown>;
        }
    } catch {
        parsedInput = {};
    }

    return {
        id: rawCall.id ?? rawCall.name ?? "function_call",
        name: rawCall.name ?? "",
        input: parsedInput,
    };
}

async function createChatCompletion(params: {
    model: string;
    messages: ChatCompletionMessage[];
    tools?: ChatCompletionTool[];
    stream?: boolean;
    maxTokens?: number;
    apiKey: string;
}): Promise<Response> {
    const response = await fetch(chatCompletionsUrl(), {
        method: "POST",
        headers: {
            Authorization: `Bearer ${params.apiKey}`,
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            model: selectedModel(params.model),
            messages: params.messages,
            tools: params.tools?.length ? params.tools : undefined,
            stream: params.stream,
            max_tokens: params.maxTokens ?? MAX_OUTPUT_TOKENS,
        }),
    });

    if (!response.ok) {
        const text = await response.text().catch(() => "");
        throw new Error(
            `OpenAI request failed (${response.status}): ${text || response.statusText}`,
        );
    }

    return response;
}

export async function streamOpenAI(
    params: StreamChatParams,
): Promise<StreamChatResult> {
    const {
        model,
        systemPrompt,
        tools = [],
        callbacks = {},
        runTools,
        apiKeys,
        enableThinking,
    } = params;
    const maxIter = params.maxIterations ?? 10;
    const key = apiKey(apiKeys?.openai);
    const responseTools = toChatCompletionTools(tools);
    const messages = toChatCompletionMessages(params.messages);
    let fullText = "";
    const hasTools = responseTools.length > 0;

    for (let iter = 0; iter < maxIter; iter++) {
        const response = await createChatCompletion({
            model,
            messages:
                iter === 0
                    ? [{ role: "system", content: systemPrompt }, ...messages]
                    : messages,
            tools: responseTools,
            stream: true,
            apiKey: key,
        });
        if (!response.body) throw new Error("OpenAI response had no body");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        const rawToolCalls = new Map<
            number,
            { id?: string; name?: string; arguments: string }
        >();
        const startedToolCallIds = new Set<string>();
        let buffer = "";
        let pendingText = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const extracted = extractSseJson(buffer);
            buffer = extracted.rest;

            for (const event of extracted.events as ChatCompletionChunk[]) {
                const delta = event.choices?.[0]?.delta;
                if (!delta) continue;

                if (typeof delta.content === "string") {
                    if (hasTools) {
                        pendingText += delta.content;
                    } else {
                        fullText += delta.content;
                        callbacks.onContentDelta?.(delta.content);
                    }
                }

                for (const partialCall of delta.tool_calls ?? []) {
                    const existing = rawToolCalls.get(partialCall.index) ?? {
                        arguments: "",
                    };
                    if (partialCall.id) existing.id = partialCall.id;
                    if (partialCall.function?.name) {
                        existing.name = partialCall.function.name;
                    }
                    if (typeof partialCall.function?.arguments === "string") {
                        existing.arguments += partialCall.function.arguments;
                    }
                    rawToolCalls.set(partialCall.index, existing);

                    const provisionalId =
                        existing.id ??
                        existing.name ??
                        `tool_call_${partialCall.index}`;
                    if (!startedToolCallIds.has(provisionalId)) {
                        startedToolCallIds.add(provisionalId);
                        const call = parseFunctionCall({
                            id: provisionalId,
                            name: existing.name,
                            arguments: existing.arguments,
                        });
                        callbacks.onToolCallStart?.(call);
                    }
                }
            }
        }

        const orderedRawCalls = [...rawToolCalls.entries()]
            .sort(([a], [b]) => a - b)
            .map(([, value], i) => ({
                id: value.id ?? value.name ?? `tool_call_${i}`,
                name: value.name ?? "",
                arguments: value.arguments,
            }));

        const toolCalls = orderedRawCalls.map((call) => parseFunctionCall(call));

        if (!toolCalls.length || !runTools) {
            if (pendingText) {
                fullText += pendingText;
                callbacks.onContentDelta?.(pendingText);
            }
            break;
        }

        const assistantToolCalls: ChatCompletionToolCall[] = orderedRawCalls.map(
            (call) => ({
                id: call.id,
                type: "function",
                function: {
                    name: call.name,
                    arguments: call.arguments || "{}",
                },
            }),
        );
        messages.push({
            role: "assistant",
            content: pendingText,
            tool_calls: assistantToolCalls,
        });

        const results = await runTools(toolCalls);
        for (const result of results) {
            messages.push({
                role: "tool",
                tool_call_id: result.tool_use_id,
                content: result.content,
            });
        }
    }

    return { fullText };
}

export async function completeOpenAIText(params: {
    model: string;
    systemPrompt?: string;
    user: string;
    maxTokens?: number;
    apiKeys?: { openai?: string | null };
}): Promise<string> {
    const response = await createChatCompletion({
        model: params.model,
        messages: [
            ...(params.systemPrompt
                ? ([{ role: "system", content: params.systemPrompt }] as const)
                : []),
            { role: "user", content: params.user },
        ],
        maxTokens: params.maxTokens ?? 512,
        apiKey: apiKey(params.apiKeys?.openai),
    });
    const json = (await response.json()) as {
        choices?: {
            message?: {
                content?: string | { type?: string; text?: string }[];
            };
        }[];
    };

    const content = json.choices?.[0]?.message?.content;
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
        return content
            .filter((part) => part?.type === "text")
            .map((part) => part.text ?? "")
            .join("");
    }
    return "";
}

export type { NormalizedToolResult };
