import { randomBytes } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { resolveDesktopAppDataDir } from "./paths";

/**
 * PERSISTED_ACCESS_TOKEN：把访问令牌持久化到用户 profile 下，并跨启动复用。
 * 原实现每次启动都生成随机令牌，外部自动化（守护脚本）无从得知真实令牌，
 * 发出的请求会被判 401 ACCESS_TOKEN_REQUIRED。
 */
const ACCESS_TOKEN_FILE_NAME = "desktop-access-token.txt";

function resolveAccessTokenFile(): string | null {
  try {
    return path.join(resolveDesktopAppDataDir(), ACCESS_TOKEN_FILE_NAME);
  }
  catch {
    return null;
  }
}

function persistAccessToken(token: string): void {
  try {
    const file = resolveAccessTokenFile();
    if (!file) {
      return;
    }
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, token, "utf8");
  }
  catch {
    // 落盘失败不影响 App 本身运行
  }
}

function resolvePersistedAccessToken(): string {
  try {
    const file = resolveAccessTokenFile();
    if (file) {
      let existing = "";
      try {
        existing = readFileSync(file, "utf8").trim();
      }
      catch {
        // 还没落盘过（首次启动），继续走生成流程。
        // 这里必须单独 try：否则 ENOENT 会跳到外层 catch，
        // 导致首次启动只返回随机令牌却从不落盘，跨启动复用失效。
      }
      if (existing) {
        return existing;
      }
    }
    const token = randomBytes(32).toString("hex");
    persistAccessToken(token);
    return token;
  }
  catch {
    return randomBytes(32).toString("hex");
  }
}

/** Managed servers and their renderer share a fresh, process-local credential. */
export function configureDesktopAccessToken(isPackaged: boolean): string | undefined {
  const mode = process.env.AI_NOVEL_DESKTOP_SERVER_MODE ?? (isPackaged ? "managed" : "external");
  const configured = process.env.AI_NOVEL_ACCESS_TOKEN?.trim();
  if (configured) {
    persistAccessToken(configured);
    return configured;
  }
  if (mode === "external") return undefined;
  const token = resolvePersistedAccessToken();
  process.env.AI_NOVEL_ACCESS_TOKEN = token;
  return token;
}
