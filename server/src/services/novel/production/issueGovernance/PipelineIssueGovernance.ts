import type { DirectorIssueCode, DirectorIssueDecision } from "@ai-novel/shared/types/directorIssue";
import type { PipelinePayload } from "../../novelCoreShared";
import { logPipelineWarn } from "../../novelCoreShared";
import {
  directorIssueService,
  type ReportDirectorIssueResult,
  type DirectorIssueTaskContext,
} from "../../director/issues";
import { isModelOutputFailure } from "../../../../modules/production/common/structuredErrors";

/**
 * 模型输出不可用（空响应、JSON 被截断、格式错误、schema 不匹配、思考内容污染等）
 * 属于"已知结果"：模型作答了，但这次调用没有留下任何持久写入，直接重试是安全的。
 *
 * 这类失败是概率性的，原先被归到 runtime.unclassified（默认动作就是挂起等人工），
 * 于是每次都要作者手动点"再继续"。现在改用目录里已有但一直没人使用的
 * generation.runtime_failed（默认动作 auto_retry），并给它一份独立的宽松预算。
 */
const MODEL_OUTPUT_ISSUE_CODE = "generation.runtime_failed";
const MODEL_OUTPUT_MAX_AUTOMATIC_RETRIES = 3;

export function resolvePipelineRuntimeIssueCode(error: unknown): DirectorIssueCode {
  if (error && typeof error === "object" && "status" in error && error.status === 402) {
    return "runtime.model_unavailable";
  }
  if (isModelOutputFailure(error)) {
    return MODEL_OUTPUT_ISSUE_CODE;
  }
  return "runtime.unclassified";
}

export async function reportPipelineIssue(input: {
  governance: DirectorIssueTaskContext | null;
  workflowTaskId?: string;
  novelId: string;
  jobId: string;
  issueCode: DirectorIssueCode;
  stage: string;
  summary: string;
  evidence?: string;
  chapterId?: string;
  chapterOrder?: number;
  qualityScores?: Record<string, number>;
  attempt?: number;
  hasUsableOutput?: boolean;
  provider?: PipelinePayload["provider"];
  model?: string;
  temperature?: number;
  applyAction?: (decision: DirectorIssueDecision) => Promise<void>;
}): Promise<ReportDirectorIssueResult | null> {
  if (!input.governance) return null;
  try {
    return await directorIssueService.reportIssue({
      issueGovernanceVersion: input.governance.issueGovernanceVersion,
      taskId: input.workflowTaskId,
      novelId: input.novelId,
      issueCode: input.issueCode,
      stage: input.stage,
      summary: input.summary,
      evidence: input.evidence,
      affectedScope: input.chapterId ? `chapter:${input.chapterId}` : `pipeline:${input.jobId}`,
      chapterId: input.chapterId,
      chapterOrder: input.chapterOrder,
      qualityScores: input.qualityScores,
      attempt: input.attempt,
      hasUsableOutput: input.hasUsableOutput,
      runMode: input.governance.runMode,
      fingerprint: [input.jobId, input.issueCode, input.chapterId ?? "book", input.attempt ?? 0].join(":"),
      // 模型输出不可用是概率性故障，走一份独立的宽松预算（3 次），
      // 不受全局策略里 maxAutomaticRetries（上限 1）的限制。
      policy: input.issueCode === MODEL_OUTPUT_ISSUE_CODE
        ? { ...input.governance.policy, maxAutomaticRetries: MODEL_OUTPUT_MAX_AUTOMATIC_RETRIES }
        : input.governance.policy,
      policySource: input.governance.policySource,
      provider: input.provider,
      model: input.model,
      temperature: input.temperature,
      applyAction: input.applyAction,
    });
  } catch (error) {
    logPipelineWarn("章节流水线问题治理失败", {
      jobId: input.jobId,
      issueCode: input.issueCode,
      error: error instanceof Error ? error.message : String(error),
    });
    throw error;
  }
}
