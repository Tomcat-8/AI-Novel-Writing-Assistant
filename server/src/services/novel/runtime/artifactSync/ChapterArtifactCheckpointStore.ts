import { prisma } from "../../../../db/prisma";

export type ChapterArtifactCheckpointClaim = "claimed" | "already_done" | "running";

export interface ChapterArtifactCheckpointIdentity {
  novelId: string;
  chapterId: string;
  contentHash: string;
  artifactType: string;
  syncMode: string;
}

export interface ChapterArtifactCheckpointRecord {
  status: string;
  metadataJson: string | null;
  updatedAt: Date;
}

const RUNNING_STALE_MS = 15 * 60 * 1000;
const MAX_FAILED_ATTEMPTS = 2;

export class ChapterArtifactCheckpointStore {
  async read(identity: ChapterArtifactCheckpointIdentity): Promise<ChapterArtifactCheckpointRecord | null> {
    return prisma.chapterArtifactSyncCheckpoint.findUnique({
      where: this.uniqueWhere(identity),
      select: { status: true, metadataJson: true, updatedAt: true },
    });
  }

  async claim(
    identity: ChapterArtifactCheckpointIdentity,
    metadata: Record<string, unknown> = {},
  ): Promise<ChapterArtifactCheckpointClaim> {
    try {
      await prisma.chapterArtifactSyncCheckpoint.create({
        data: {
          ...identity,
          status: "running",
          sourceType: "chapter_background_sync",
          sourceStage: "chapter_execution",
          metadataJson: JSON.stringify(metadata),
        },
      });
      return "claimed";
    } catch (createError) {
      const existing = await this.read(identity).catch(() => null);
      if (!existing) {
        throw createError;
      }
      if (existing.status === "succeeded") {
        return "already_done";
      }
      let retryMetadata = metadata;
      if (existing.status === "failed") {
        // 失败检查点必须能重新认领重跑。
        // 章节资产抽取失败是概率事件（模型输出不可用、余额不足、网络抖动），
        // 真正的重试上限由调用方（导演自动重试预算）控制。
        // 这里返回 already_done 会让整章永久卡死：作者手动点"从保存进度继续"
        // 也救不回来——失败检查点里根本没有可解析的抽取结果，readExtraction
        // 会抛"章节资产抽取检查点版本不受支持。"。
        // 超过上限只重置计数，不阻断重跑。
        let failedMetadata: Record<string, unknown> = {};
        try {
          failedMetadata = existing.metadataJson ? JSON.parse(existing.metadataJson) as Record<string, unknown> : {};
        } catch {
          failedMetadata = {};
        }
        const attemptCount = Number(failedMetadata.attemptCount ?? 0);
        retryMetadata = {
          ...failedMetadata,
          ...retryMetadata,
          attemptCount: attemptCount >= MAX_FAILED_ATTEMPTS ? 1 : attemptCount + 1,
          reason: undefined,
        };
      }
      const staleBefore = new Date(Date.now() - RUNNING_STALE_MS);
      if (existing.status === "running" && existing.updatedAt > staleBefore) {
        return "running";
      }
      const claimed = await prisma.chapterArtifactSyncCheckpoint.updateMany({
        where: {
          ...identity,
          OR: [
            { status: { not: "running" } },
            { updatedAt: { lt: staleBefore } },
          ],
        },
        data: {
          status: "running",
          sourceType: "chapter_background_sync",
          sourceStage: "chapter_execution",
          metadataJson: JSON.stringify(retryMetadata),
          updatedAt: new Date(),
        },
      });
      return claimed.count > 0 ? "claimed" : "running";
    }
  }

  async succeed(
    identity: ChapterArtifactCheckpointIdentity,
    metadata: Record<string, unknown>,
  ): Promise<void> {
    await prisma.chapterArtifactSyncCheckpoint.upsert({
      where: this.uniqueWhere(identity),
      create: {
        ...identity,
        status: "succeeded",
        sourceType: "chapter_background_sync",
        sourceStage: "chapter_execution",
        metadataJson: JSON.stringify(metadata),
      },
      update: {
        status: "succeeded",
        sourceType: "chapter_background_sync",
        sourceStage: "chapter_execution",
        metadataJson: JSON.stringify(metadata),
        updatedAt: new Date(),
      },
    });
  }

  async fail(
    identity: ChapterArtifactCheckpointIdentity,
    error: unknown,
  ): Promise<void> {
    await prisma.chapterArtifactSyncCheckpoint.updateMany({
      where: { ...identity, status: "running" },
      data: {
        status: "failed",
        metadataJson: JSON.stringify({
          attemptCount: await this.readAttemptCount(identity),
          reason: error instanceof Error ? error.message : String(error),
        }),
        updatedAt: new Date(),
      },
    });
  }

  private async readAttemptCount(identity: ChapterArtifactCheckpointIdentity): Promise<number> {
    const row = await this.read(identity).catch(() => null);
    if (!row?.metadataJson) return 1;
    try {
      const metadata = JSON.parse(row.metadataJson) as Record<string, unknown>;
      return Number(metadata.attemptCount ?? 0) + 1;
    } catch {
      return 1;
    }
  }

  private uniqueWhere(identity: ChapterArtifactCheckpointIdentity) {
    return {
      novelId_chapterId_contentHash_artifactType_syncMode: identity,
    };
  }
}
