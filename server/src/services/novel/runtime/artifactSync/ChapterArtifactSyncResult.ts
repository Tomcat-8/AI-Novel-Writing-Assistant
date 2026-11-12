export type ChapterArtifactSyncStatus = "completed" | "pending" | "degraded" | "failed";

export interface ChapterArtifactSyncResult {
  status: ChapterArtifactSyncStatus;
  contentHash: string;
  completedArtifacts: string[];
  reason?: string;
  characterReviewId?: string;
  /**
   * 失败类别（如 "malformed_json" / "empty_response"）。
   * 上层据此区分"模型输出不可用"（概率性故障，可自动重试）
   * 与"数据完整性风险"（必须挂起等人工）。
   */
  failureCategory?: string | null;
}

export function mergeChapterArtifactSyncResults(
  contentHash: string,
  ...results: ChapterArtifactSyncResult[]
): ChapterArtifactSyncResult {
  const status = results.some((result) => result.status === "failed")
    ? "failed"
    : results.some((result) => result.status === "pending")
      ? "pending"
      : results.some((result) => result.status === "degraded")
        ? "degraded"
        : "completed";
  return {
    status,
    contentHash,
    completedArtifacts: [...new Set(results.flatMap((result) => result.completedArtifacts))],
    reason: results.find((result) => result.status === status)?.reason,
    characterReviewId: results.find(result => result.characterReviewId)?.characterReviewId,
    failureCategory: results.find((result) => result.failureCategory)?.failureCategory ?? null,
  };
}

export class CharacterCandidateReviewRequiredError extends Error {
  constructor(readonly reviewId: string) {
    super("本章正文已保存，请确认本章人物后继续；可选择新增、关联已有角色或忽略。");
    this.name = "CharacterCandidateReviewRequiredError";
  }
}

export class ChapterArtifactSyncBoundaryError extends Error {
  constructor(readonly result: ChapterArtifactSyncResult) {
    super(result.reason || `章节资产同步未完成（${result.status}）。`);
    this.name = "ChapterArtifactSyncBoundaryError";
  }
}

/** A saved chapter changed while a background artifact run was still active. */
export class ChapterArtifactContentVersionError extends Error {
  constructor(message = "章节正文版本已变化，已拒绝过期资产结果。") {
    super(message);
    this.name = "ChapterArtifactContentVersionError";
  }
}
