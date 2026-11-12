/**
 * 把「顶部全局模型选择」同步到正在运行的导演任务。
 *
 * 背景（2026-11-11）：
 *   - 顶部模型选择器写的是 AppSetting 的 `llm.currentSelection`（PUT /api/settings/llm-selection）。
 *   - 但导演任务的模型是**冻结**在 `DirectorNextRun.contractJson` 里的
 *     （modelConfig.route / modelConfig.model），跟系统设置里的 11 条任务路由无关。
 *   - 正文流水线启动时把 provider/model 缓存进 runtimePayload，整批章节都用它，
 *     所以只改数据库对**正在跑**的作业没有任何影响。
 *
 * 做法：
 *   1. 同步 `DirectorNextRun.contractJson`（modelConfig + launchInput）—— 这是事实来源，
 *      流水线每个章节边界会重新从契约解析模型。
 *   2. 同步该 run 绑定作业 `GenerationJob.payload` 的 provider/model —— 让中途崩溃/续跑也保持一致。
 *   3. 同步该小说绑定的导演任务 seed —— 让「从保存进度继续」与界面表单显示保持一致。
 *   4. 追加一条 `model_switched` 事件，运行记录里能看到切换痕迹。
 *
 * 只同步「正在跑」的 run；已暂停 / 已完成 / 已取消的保持原模型。
 */
import { prisma } from "../../../../db/prisma";

/** 只有正在跑的导演任务才跟随全局模型切换。 */
export const DIRECTOR_RUN_MODEL_SYNC_TARGET_STATUS = ["running"] as const;

export interface DirectorRunModelSyncResult {
  runId: string;
  novelId: string;
  jobId: string | null;
  from: { provider: string | null; model: string | null };
  to: { provider: string; model: string };
  jobUpdated: boolean;
}

export interface DirectorRunModelSyncInput {
  provider?: unknown;
  model?: unknown;
}

interface DirectorRunContract {
  modelConfig?: { route?: unknown; model?: unknown };
  launchInput?: Record<string, unknown>;
}

type PrismaClient = typeof prisma;

function parseJsonObject(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

function normalizeProvider(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function normalizeModel(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

async function readBoundJobId(db: PrismaClient, runId: string): Promise<string | null> {
  const event = await db.directorNextEvent.findFirst({
    where: { runId, type: "chapter_batch_job" },
    orderBy: { seq: "desc" },
    select: { payloadJson: true },
  });
  const jobId = parseJsonObject(event?.payloadJson)?.jobId;
  return typeof jobId === "string" && jobId.trim() ? jobId.trim() : null;
}

async function updateJobModel(
  db: PrismaClient,
  jobId: string,
  provider: string,
  model: string,
): Promise<boolean> {
  const job = await db.generationJob.findUnique({ where: { id: jobId }, select: { payload: true } });
  if (!job) {
    return false;
  }
  const payload = parseJsonObject(job.payload);
  if (!payload) {
    return false;
  }
  if (payload.provider === provider && payload.model === model) {
    return false;
  }
  payload.provider = provider;
  payload.model = model;
  await db.generationJob.update({ where: { id: jobId }, data: { payload: JSON.stringify(payload) } });
  return true;
}

async function patchTaskSeed(
  db: PrismaClient,
  novelId: string,
  provider: string,
  model: string,
): Promise<number> {
  const rows = await db.novelWorkflowTask.findMany({
    where: { novelId },
    select: { id: true, seedPayloadJson: true },
  });
  let changed = 0;
  for (const row of rows) {
    const seed = parseJsonObject(row.seedPayloadJson);
    if (!seed) {
      continue;
    }
    seed.provider = provider;
    seed.model = model;
    const directorInput = seed.directorInput;
    if (directorInput && typeof directorInput === "object" && !Array.isArray(directorInput)) {
      seed.directorInput = { ...(directorInput as Record<string, unknown>), provider, model };
    }
    await db.novelWorkflowTask.update({
      where: { id: row.id },
      data: { seedPayloadJson: JSON.stringify(seed) },
    });
    changed += 1;
  }
  return changed;
}

async function appendModelSwitchEvent(
  db: PrismaClient,
  runId: string,
  payload: Record<string, unknown>,
): Promise<number> {
  const latest = await db.directorNextEvent.findFirst({
    where: { runId },
    orderBy: { seq: "desc" },
    select: { seq: true },
  });
  const seq = (latest?.seq ?? 0) + 1;
  await db.directorNextEvent.create({
    data: {
      id: `${runId}:${seq}`,
      runId,
      seq,
      type: "model_switched",
      payloadJson: JSON.stringify(payload),
    },
  });
  return seq;
}

/**
 * 把全局模型选择同步到所有正在运行的导演任务。
 */
export async function syncRunningRunsToModel(
  input: DirectorRunModelSyncInput,
): Promise<DirectorRunModelSyncResult[]> {
  const provider = normalizeProvider(input?.provider);
  const model = normalizeModel(input?.model);
  if (!provider || !model) {
    throw new Error("模型厂商和模型名称不能为空。");
  }
  const db = prisma;
  const controls = await db.directorNextRunControl.findMany({
    where: { status: { in: [...DIRECTOR_RUN_MODEL_SYNC_TARGET_STATUS] } },
    select: { runId: true },
  });
  const runIds = controls.map((row) => row.runId).filter((id) => typeof id === "string" && id.trim());
  if (runIds.length === 0) {
    return [];
  }
  const runs = await db.directorNextRun.findMany({
    where: { id: { in: runIds } },
    select: { id: true, novelId: true, contractJson: true },
  });
  const synced: DirectorRunModelSyncResult[] = [];
  for (const run of runs) {
    const contract = parseJsonObject(run.contractJson) as DirectorRunContract | null;
    if (!contract) {
      continue;
    }
    const from = {
      provider: normalizeProvider(contract.modelConfig?.route),
      model: normalizeProvider(contract.modelConfig?.model),
    };
    if (from.provider === provider && from.model === model) {
      continue;
    }
    const modelConfig = { ...(contract.modelConfig ?? {}), route: provider, model };
    const launchInput = { ...(contract.launchInput ?? {}), provider, model };
    await db.directorNextRun.update({
      where: { id: run.id },
      data: { contractJson: JSON.stringify({ ...contract, modelConfig, launchInput }) },
    });
    const jobId = await readBoundJobId(db, run.id);
    let jobUpdated = false;
    if (jobId) {
      jobUpdated = await updateJobModel(db, jobId, provider, model);
    }
    await patchTaskSeed(db, run.novelId, provider, model);
    await appendModelSwitchEvent(db, run.id, {
      from,
      to: { provider, model },
      jobId,
      jobUpdated,
      at: new Date().toISOString(),
    });
    synced.push({ runId: run.id, novelId: run.novelId, jobId, from, to: { provider, model }, jobUpdated });
  }
  return synced;
}
