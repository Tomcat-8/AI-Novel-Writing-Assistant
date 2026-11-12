/**
 * Structured-output failures whose outcome is KNOWN: the model answered, but the
 * output could not be used (unparsable, incomplete, schema-violating, polluted by
 * reasoning, or lacking native JSON support). The model port performed no durable
 * write for such a call, so releasing its claim and retrying is safe.
 *
 * Everything else stays unknown: transport_error means the request may or may not
 * have executed, and any other error may hide a side effect. Claims for those
 * outcomes must remain claimed, exactly as before.
 *
 * The check reads the public `category` contract of StructuredOutputError
 * (server/src/llm/structuredOutput.ts) without importing the LLM layer, so the
 * production module keeps its dependency direction. The category values are the
 * same ones the rest of the codebase already matches on directly.
 */
const RETRYABLE_STRUCTURED_OUTPUT_CATEGORIES = new Set([
  "malformed_json",
  "schema_mismatch",
  "incomplete_json",
  "thinking_pollution",
  "unsupported_native_json",
  "empty_response",
  "reasoning_budget_exhausted",
  "output_budget_exhausted",
]);

export function isRetryableModelOutputError(error: unknown): boolean {
  if (typeof error !== "object" || error === null) return false;
  const category = (error as { category?: unknown }).category;
  return typeof category === "string" && RETRYABLE_STRUCTURED_OUTPUT_CATEGORIES.has(category);
}

/**
 * A semantic (post-validation) rejection is the same kind of known outcome:
 * the model answered, the output was fully received, assessed and rejected
 * (for example a contract that duplicates the previous chapter), and the
 * model port performed no durable write for the call. Releasing its claim
 * and retrying is safe.
 *
 * The check reads the public `promptQualityFailureKind` marker the prompt
 * runner attaches (server/src/prompting/core/promptRunner.ts) without
 * importing the prompting layer. Only the explicit `post_validate_failed`
 * mark counts: transport and other failures are never marked this way.
 */
export function isKnownPromptQualityFailure(error: unknown): boolean {
  if (typeof error !== "object" || error === null) return false;
  return (error as { promptQualityFailureKind?: unknown }).promptQualityFailureKind === "post_validate_failed";
}

/**
 * 结构化输出失败的类别全集：模型这次没有产出可用的结果。
 *
 * 与 isRetryableModelOutputError 的分工不同：前者回答"这次调用有没有留下持久写入"
 * （决定能不能释放检查点占位），这里回答"这次失败是不是模型输出不可用"
 * （决定要不要按概率性故障自动重试，而不是立刻挂起等人工）。
 *
 * transport_error 也计入这一类：章节正文没有落库，重试不会造成重复写入。
 */
const MODEL_OUTPUT_FAILURE_CATEGORIES = new Set([
  "malformed_json",
  "schema_mismatch",
  "incomplete_json",
  "thinking_pollution",
  "unsupported_native_json",
  "empty_response",
  "transport_error",
]);

/**
 * 判断一个错误是不是"模型作答了但输出不可用"这类概率性故障，并返回类别。
 *
 * 依次检查四处，避免错误被中间层包装后丢掉标记：
 * 1. StructuredOutputError 的公开 category 契约；
 * 2. 章节资产同步的边界错误把失败类别带在 result.failureCategory 上；
 * 3. 提示词运行器附加的 promptQualityFailureKind（语义校验失败）；
 * 4. 错误消息里的 [STRUCTURED_OUTPUT:xxx] 标记（category 被包装丢失时的兜底）。
 */
export function modelOutputFailureCategory(error: unknown): string | null {
  if (typeof error !== "object" || error === null) {
    return null;
  }
  const candidate = error as {
    category?: unknown;
    promptQualityFailureKind?: unknown;
    message?: unknown;
    result?: unknown;
  };
  if (typeof candidate.category === "string" && MODEL_OUTPUT_FAILURE_CATEGORIES.has(candidate.category)) {
    return candidate.category;
  }
  // 章节资产同步的边界错误会把失败类别带在 result.failureCategory 上。
  const boundaryCategory = candidate.result && typeof candidate.result === "object"
    ? (candidate.result as { failureCategory?: unknown }).failureCategory
    : undefined;
  if (typeof boundaryCategory === "string" && boundaryCategory.length > 0) {
    return boundaryCategory;
  }
  if (candidate.promptQualityFailureKind === "post_validate_failed") {
    return "post_validate_failed";
  }
  const message = typeof candidate.message === "string" ? candidate.message : "";
  const matched = message.match(/\[STRUCTURED_OUTPUT:([a-z_]+)\]/i);
  if (matched) {
    const category = matched[1].toLowerCase();
    if (MODEL_OUTPUT_FAILURE_CATEGORIES.has(category)) {
      return category;
    }
  }
  return null;
}

export function isModelOutputFailure(error: unknown): boolean {
  return modelOutputFailureCategory(error) !== null;
}
