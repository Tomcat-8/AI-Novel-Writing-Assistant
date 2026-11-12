import { recordInvocationParseFailure } from "./compatibility";
import { classifyResponseEvidence, terminalResponseReason } from "../platform/llm/diagnostics/structuredEvidence";
import type { StructuredResponseEvidence } from "@ai-novel/shared/types/structuredDiagnostics";
import { z, type ZodError, type ZodType } from "zod";
import type { LLMProvider } from "@ai-novel/shared/types/llm";
import type { ModelRouteRequestProtocol } from "@ai-novel/shared/types/novel";
import type { TaskType } from "./modelRouter";
import { relaxGeneratedContentSchema } from "./generatedContentSchema";
import { repairWithLlm } from "./structuredInvokeRepair";
import {
  classifyStructuredOutputFailure,
  resolveStructuredOutputProfile,
  schemaAllowsTopLevelArray,
  selectStructuredOutputStrategy,
  describeStructuredResponseFailure,
  StructuredOutputError,
  type StructuredOutputDiagnostics,
  type StructuredOutputErrorCategory,
  type StructuredOutputProfile,
  type StructuredOutputStrategy,
} from "./structuredOutput";
import { extractJSONValue } from "../services/novel/novelP0Utils";
import type { PromptInvocationMeta } from "../prompting/core/promptTypes";
import type { LlmTokenUsageSnapshot } from "./usageTracking";

export interface StructuredInvokeResult<T> {
  data: T;
  repairUsed: boolean;
  repairAttempts: number;
  diagnostics: StructuredOutputDiagnostics;
  tokenUsage?: LlmTokenUsageSnapshot | null;
}

export interface StructuredInvokeRawParseInput<T> {
  diagnosticContext?: import("../platform/llm/diagnostics/StructuredDiagnosticStore").StructuredDiagnosticContext;
  rawContent: string;
  responseEvidence?: StructuredResponseEvidence;
  schema: ZodType<T>;
  tokenUsage?: LlmTokenUsageSnapshot | null;
  provider?: LLMProvider;
  model?: string;
  apiKey?: string;
  baseURL?: string;
  temperature?: number;
  maxTokens?: number;
  timeoutMs?: number;
  signal?: AbortSignal;
  taskType?: TaskType;
  requestProtocol?: ModelRouteRequestProtocol;
  label: string;
  maxRepairAttempts?: number;
  repairGuidance?: string;
  promptMeta?: PromptInvocationMeta;
  onRepairOutputDelta?: (content: string) => void;
  repairMessages?: import("@langchain/core/messages").BaseMessage[];
  strategy: StructuredOutputStrategy;
  profile: StructuredOutputProfile;
  fallbackAvailable?: boolean;
  fallbackUsed?: boolean;
  reasoningForcedOff?: boolean;
}

function tryFixTruncatedJson(raw: string): string {
  const text = raw.trim();
  if (!text) return text;

  const count = (re: RegExp) => (text.match(re) ?? []).length;
  const openBraces = count(/{/g);
  const closeBraces = count(/}/g);
  const openBrackets = count(/\[/g);
  const closeBrackets = count(/]/g);

  let fixed = text.replace(/,\s*$/g, "");
  if (openBrackets > closeBrackets) {
    fixed += "]".repeat(openBrackets - closeBrackets);
  }
  if (openBraces > closeBraces) {
    fixed += "}".repeat(openBraces - closeBraces);
  }
  return fixed;
}

// 结构化输出的 JSON 容错修补。
// 模型偶发两类小毛病：对象里漏一个逗号（"Expected ',' or '}' after property value"），
// 以及字符串里塞裸换行/制表符。两者都会让整章抽取判失败、流水线暂停。
// 这里在 JSON.parse 失败后做一次修补，修不好仍然按原逻辑报错。
function escapeJsonStringControlChars(text: string): string {
  let out = "";
  let inString = false;
  let escaped = false;
  for (const ch of text) {
    if (inString) {
      if (escaped) {
        out += ch;
        escaped = false;
        continue;
      }
      if (ch === "\\") {
        out += ch;
        escaped = true;
        continue;
      }
      if (ch === "\"") {
        out += ch;
        inString = false;
        continue;
      }
      if (ch === "\n") {
        out += "\\n";
        continue;
      }
      if (ch === "\r") {
        out += "\\r";
        continue;
      }
      if (ch === "\t") {
        out += "\\t";
        continue;
      }
      if (ch.charCodeAt(0) < 32) {
        out += "\\u" + ch.charCodeAt(0).toString(16).padStart(4, "0");
        continue;
      }
      out += ch;
      continue;
    }
    if (ch === "\"") {
      inString = true;
    }
    out += ch;
  }
  return out;
}

function findJsonStringEnd(text: string, start: number): number {
  let index = start + 1;
  let escaped = false;
  while (index < text.length) {
    const ch = text[index];
    if (escaped) {
      escaped = false;
    } else if (ch === "\\") {
      escaped = true;
    } else if (ch === "\"") {
      return index;
    }
    index += 1;
  }
  return -1;
}

function repairJsonPunctuation(text: string): string {
  let out = "";
  let index = 0;
  let inString = false;
  let escaped = false;
  let valueEnd = false;
  let expectingValue = false;
  const stack: string[] = [];
  const pushComma = () => {
    out += ",";
  };
  while (index < text.length) {
    const ch = text[index];
    if (inString) {
      if (escaped) {
        out += ch;
        escaped = false;
        index += 1;
        continue;
      }
      if (ch === "\\") {
        out += ch;
        escaped = true;
        index += 1;
        continue;
      }
      if (ch === "\"") {
        out += ch;
        inString = false;
        valueEnd = true;
        expectingValue = false;
        index += 1;
        continue;
      }
      out += ch;
      index += 1;
      continue;
    }
    if (ch === "\"") {
      // 刚读完一个值，紧接着又来一个字符串：对象里多半是漏了逗号的 key。
      if (valueEnd && !expectingValue && stack[stack.length - 1] === "{") {
        const keyEnd = findJsonStringEnd(text, index);
        if (keyEnd > 0) {
          let probe = keyEnd + 1;
          while (probe < text.length && /\s/.test(text[probe])) {
            probe += 1;
          }
          if (text[probe] === ":") {
            pushComma();
          }
        }
      } else if (valueEnd && !expectingValue && stack[stack.length - 1] === "[") {
        pushComma();
      }
      out += ch;
      inString = true;
      index += 1;
      continue;
    }
    if (ch === "{" || ch === "[") {
      if (valueEnd && !expectingValue && stack[stack.length - 1] === "[") {
        pushComma();
      }
      out += ch;
      stack.push(ch);
      valueEnd = false;
      expectingValue = false;
      index += 1;
      continue;
    }
    if (ch === "}" || ch === "]") {
      out += ch;
      if (stack.length > 0) {
        stack.pop();
      }
      valueEnd = true;
      expectingValue = false;
      index += 1;
      continue;
    }
    if (ch === ":") {
      out += ch;
      valueEnd = false;
      expectingValue = true;
      index += 1;
      continue;
    }
    if (ch === ",") {
      out += ch;
      valueEnd = false;
      expectingValue = true;
      index += 1;
      continue;
    }
    if (/\s/.test(ch)) {
      out += ch;
      index += 1;
      continue;
    }
    if (/[-0-9tfn]/.test(ch)) {
      // 数字 / true / false / null 整体当成一个字面量
      if (valueEnd && !expectingValue && stack[stack.length - 1] === "[") {
        pushComma();
      }
      let probe = index;
      while (probe < text.length && /[-+0-9a-zA-Z_.eE]/.test(text[probe])) {
        probe += 1;
      }
      out += text.slice(index, probe);
      valueEnd = true;
      expectingValue = false;
      index = probe;
      continue;
    }
    out += ch;
    valueEnd = false;
    expectingValue = false;
    index += 1;
  }
  return out;
}

function markJsonStringPositions(text: string): boolean[] {
  const flags: boolean[] = new Array(text.length).fill(false);
  let inString = false;
  let escaped = false;
  for (let index = 0; index < text.length; index += 1) {
    const ch = text[index];
    if (inString) {
      flags[index] = true;
      if (escaped) {
        escaped = false;
      } else if (ch === "\\") {
        escaped = true;
      } else if (ch === "\"") {
        inString = false;
      }
      continue;
    }
    if (ch === "\"") {
      inString = true;
      flags[index] = true;
    }
  }
  return flags;
}

function findJsonKeyEnd(text: string, index: number): number {
  const end = findJsonStringEnd(text, index);
  if (end < 0) {
    return -1;
  }
  let probe = end + 1;
  while (probe < text.length && /\s/.test(text[probe])) {
    probe += 1;
  }
  return text[probe] === ":" ? end : -1;
}

/**
 * 计算 text[0, position) 区间里「未闭合的数组」层数（字符串内的括号不计）。
 */
function jsonArrayDepthAt(text: string, position: number): number {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = 0; index < position; index += 1) {
    const ch = text[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === "\\") {
      escaped = true;
      continue;
    }
    if (ch === "\"") {
      inString = !inString;
      continue;
    }
    if (inString) {
      continue;
    }
    if (ch === "[") {
      depth += 1;
    } else if (ch === "]") {
      depth -= 1;
    }
  }
  return depth;
}

/**
 * REPAIR_STRUCTURAL_SHAPE（2026-11-12）：修复「模型把 JSON 局部写坏」的第五类畸形。
 *
 * 真实样本（260 章跑完后从日志统计，114 条失败里除截断外还有 22 条）：
 *   1) 缺冒号        "currentFact","沈砚…"              → Expected ':' after property name
 *   2) 多余的 key    "kind":"present","mention","summary"
 *   3) key 缺左引号  …, mustAdvance": [                 → Expected double-quoted property name
 *   4) 杂散标点      …, - "issueIds": [  /  …},*,"x":    → Expected double-quoted property name
 *   5) 杂散括号      …}]}],  /  …]},                    → Expecting property name
 *   6) 小数点变逗号  "verifiedCount":0,375}             → Expected double-quoted property name
 *   7) key 被转义    …},\"characterResourceDeltas"        → Expected double-quoted property name
 *   8) 数组里写成对象 {"evidence":["…","confidence":0.95]} → Expected ',' or ']'
 *   9) 漏逗号        {"a":1}{"b":2}                     → Expected ',' or ']'
 *  10) 收尾多余逗号  …"targetEndChapterOrder":181]},}    → Expected double-quoted property name
 *  11) 多余的 "["     …"reveals":[…],["riskNotes":[…]    → Expected double-quoted property name
 *  12) 丢了 key      …"stateDeltas":{…}},[{"resourceName":…}
 *                                                   → Expected double-quoted property name
 *
 * 老的位置式修补修不了这些：
 *   - 「裸 key」只认 `key:`，认不了 `key":`（缺左引号）；
 *   - 「漏逗号」要求补完后整段能 JSON.parse 成功，可模型常常同时犯 2~3 个错，
 *     整段永远解析不成功 → 这一轮直接放弃，12 轮机会白白浪费。
 *
 * 注意：V8 报的 position 指向「让它失败的那个 token」，不一定是真正的病灶，
 *       所以每条规则都要看 position 附近的上下文再决定怎么改，不能只信 position。
 */

/** zod v4 的内部结构探针：只用到 type / shape / element / innerType / in / out。 */
interface ZodShapeDef {
  type?: string;
  shape?: Record<string, unknown> | (() => Record<string, unknown>);
  element?: unknown;
  innerType?: unknown;
  out?: unknown;
  in?: unknown;
}

/**
 * REPAIR_STRUCTURAL_SHAPE：解析期间临时挂上 zod schema。
 *
 * 结构化解析链路全是同步调用，所以用「进入前切换、退出后还原」的写法即可，
 * 并发请求之间不会串味 —— 比把 schema 一路透传下去改动小得多。
 */
let __aiNovelStructuredSchemaHint: unknown = null;

function withSchemaHint<T>(schema: unknown, run: () => T): T {
  const previous = __aiNovelStructuredSchemaHint;
  __aiNovelStructuredSchemaHint = schema ?? null;
  try {
    return run();
  } finally {
    __aiNovelStructuredSchemaHint = previous;
  }
}

function zodDefOf(value: unknown): ZodShapeDef | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const holder = value as { def?: unknown; _def?: unknown };
  const def = holder.def ?? holder._def;
  return def && typeof def === "object" ? def as ZodShapeDef : null;
}

function zodShapeOf(value: unknown): Record<string, unknown> | null {
  const def = zodDefOf(value);
  if (!def) {
    return null;
  }
  const shape = def.shape;
  const resolved = typeof shape === "function" ? shape() : shape;
  return resolved && typeof resolved === "object" ? resolved : null;
}

/** 拆掉 optional / nullable / default / pipe 等外壳，返回内部的对象 schema。 */
function unwrapToObject(value: unknown): unknown {
  let current: unknown = value;
  for (let guard = 0; guard < 16; guard += 1) {
    const def = zodDefOf(current);
    if (!def) {
      return null;
    }
    const type = def.type;
    if (type === "object") {
      return current;
    }
    if (type === "pipe") {
      // z.preprocess(fn, schema)：def.in 是变换，def.out 才是真正的 schema
      current = def.out ?? def.in;
      continue;
    }
    if (type === "optional" || type === "nullable" || type === "default"
      || type === "prefault" || type === "catch" || type === "readonly"
      || type === "nonoptional") {
      current = def.innerType;
      continue;
    }
    return null;
  }
  return null;
}

function objectShapeKeysOf(value: unknown): string[] | null {
  const objectSchema = unwrapToObject(value);
  if (!objectSchema) {
    return null;
  }
  const shape = zodShapeOf(objectSchema);
  return shape ? Object.keys(shape) : null;
}

/**
 * 拆掉 optional / nullable / default 等外壳，返回内部的数组 schema。
 * 注意：artifact_delta 的 characterResourceDeltas 是 z.array(...).max(8).default([])，
 *       外层 def.type 是 "default" 而不是 "array" —— 不拆外壳就永远认不出来。
 */
function unwrapToArray(value: unknown): unknown {
  let current: unknown = value;
  for (let guard = 0; guard < 16; guard += 1) {
    const def = zodDefOf(current);
    if (!def) {
      return null;
    }
    const type = def.type;
    if (type === "array") {
      return current;
    }
    if (type === "optional" || type === "nullable" || type === "default"
      || type === "prefault" || type === "catch" || type === "readonly"
      || type === "nonoptional") {
      current = def.innerType;
      continue;
    }
    return null;
  }
  return null;
}

function arrayElementShapeKeysOf(value: unknown): string[] | null {
  const arraySchema = unwrapToArray(value);
  if (!arraySchema) {
    return null;
  }
  const def = zodDefOf(arraySchema);
  return objectShapeKeysOf(def?.element);
}

interface JsonContainerFrame {
  type: string;
  key: string | null;
}

/** 找到 index 所在的那个对象，在根对象里的属性路径（属性名数组）。 */
function containerPathAt(value: string, index: number): JsonContainerFrame[] | null {
  const stack: JsonContainerFrame[] = [];
  let pendingKey: string | null = null;
  for (let i = 0; i < index; i += 1) {
    const ch = value[i];
    if (ch === "\"") {
      const end = findJsonStringEnd(value, i);
      if (end < 0) {
        return null;
      }
      const token = value.slice(i, end + 1);
      const afterToken = value.slice(end + 1).replace(/^\s+/, "");
      if (afterToken.startsWith(":")) {
        try {
          pendingKey = JSON.parse(token) as string;
        } catch {
          return null;
        }
      }
      i = end;
      continue;
    }
    if (ch === "{" || ch === "[") {
      stack.push({ type: ch, key: pendingKey });
      pendingKey = null;
      continue;
    }
    if (ch === "}" || ch === "]") {
      stack.pop();
      pendingKey = null;
      continue;
    }
    if (ch === ",") {
      pendingKey = null;
    }
  }
  return stack;
}

/** 取 index 处那个对象的**第一层**字段名（只取深度 1，嵌套里的字段不算）。 */
function collectFirstObjectKeys(value: string, index: number): string[] | null {
  let i = index;
  while (i < value.length && /\s/.test(value[i])) {
    i += 1;
  }
  if (value[i] !== "{") {
    return null;
  }
  const keys: string[] = [];
  let depth = 0;
  for (; i < value.length; i += 1) {
    const ch = value[i];
    if (ch === "\"") {
      const end = findJsonStringEnd(value, i);
      if (end < 0) {
        return null;
      }
      const token = value.slice(i, end + 1);
      const afterToken = value.slice(end + 1).replace(/^\s+/, "");
      if (depth === 1 && afterToken.startsWith(":")) {
        try {
          keys.push(JSON.parse(token) as string);
        } catch {
          return null;
        }
      }
      i = end;
      continue;
    }
    if (ch === "{" || ch === "[") {
      depth += 1;
      continue;
    }
    if (ch === "}" || ch === "]") {
      depth -= 1;
      if (depth === 0) {
        return keys;
      }
    }
  }
  return null;
}

/**
 * 按 schema 反推丢掉的 key 名：
 *   找到 position 所在的对象 → 顺着属性路径找到它在 schema 里的对应对象 →
 *   遍历该对象的属性，凡是「数组且元素是对象」的用元素字段签名打分，
 *   得分最高且唯一的就是被吞掉的那个 key。
 */
function inferMissingKeyFromSchema(schema: unknown, value: string, index: number): string | null {
  if (!schema) {
    return null;
  }
  const rootObject = unwrapToObject(schema);
  if (!rootObject) {
    return null;
  }
  const stack = containerPathAt(value, index);
  if (!stack || stack.length === 0) {
    return null;
  }
  const top = stack[stack.length - 1];
  if (top.type !== "{") {
    return null;
  }
  let currentObject: unknown = rootObject;
  for (let frameIndex = 0; frameIndex < stack.length; frameIndex += 1) {
    const frame = stack[frameIndex];
    if (frame.type !== "{") {
      // 数组里的对象：属性路径断了，反推不了
      return null;
    }
    if (frameIndex === 0) {
      if (frame.key !== null) {
        return null;
      }
      continue;
    }
    if (!frame.key) {
      return null;
    }
    const shape = zodShapeOf(currentObject);
    if (!shape) {
      return null;
    }
    const next = shape[frame.key];
    if (!next) {
      return null;
    }
    const unwrapped = unwrapToObject(next);
    if (!unwrapped) {
      return null;
    }
    currentObject = unwrapped;
  }
  const shape = zodShapeOf(currentObject);
  if (!shape) {
    return null;
  }
  // 孤儿值：数组 → 取第一个对象的字段；对象 → 直接取字段
  const isArray = value[index] === "[";
  const orphanKeys = collectFirstObjectKeys(value, isArray ? index + 1 : index);
  if (!orphanKeys || orphanKeys.length === 0) {
    return null;
  }
  const orphanSet = new Set(orphanKeys);
  let bestKey: string | null = null;
  let bestScore = 0;
  let tied = false;
  for (const key of Object.keys(shape)) {
    const candidateKeys = isArray
      ? arrayElementShapeKeysOf(shape[key])
      : objectShapeKeysOf(shape[key]);
    if (!candidateKeys || candidateKeys.length === 0) {
      continue;
    }
    let score = 0;
    for (const orphanKey of orphanSet) {
      if (candidateKeys.includes(orphanKey)) {
        score += 1;
      }
    }
    if (score > bestScore) {
      bestScore = score;
      bestKey = key;
      tied = false;
    } else if (score === bestScore && score > 0) {
      tied = true;
    }
  }
  // 至少要命中 2 个字段，而且必须唯一 —— 否则宁可不修
  if (!bestKey || bestScore < 2 || tied) {
    return null;
  }
  return bestKey;
}

/** 单轮外科式修补：只改一处，返回修补后的文本；不改则返回 null。 */
function applyJsonStructuralFixRound(text: string, error: unknown): string | null {
  const message = error instanceof Error ? error.message : String(error ?? "");
  const matched = /at position (\d+)/.exec(message);
  if (!matched) {
    return null;
  }
  const position = Number(matched[1]);
  // 注意：position 可以等于 text.length —— 解析器「一路读到文本末尾」时报错就是这种情况，
  // 而末尾漏收尾括号恰恰要靠这个位置来补。所以这里用 > 而不是 >=。
  if (position < 0 || position > text.length) {
    return null;
  }
  const skipWs = (value: string): string => value.replace(/^\s+/, "");
  const trimmedBefore = text.slice(0, position).replace(/\s+$/, "");
  const after = text.slice(position);
  const afterNoWs = skipWs(after);
  const nextChar = afterNoWs.slice(0, 1);
  // lastChar = position 前面那个非空白字符；prevChar = 再前面一个
  const lastChar = trimmedBefore.slice(-1);
  const prevChar = trimmedBefore.slice(0, -1).replace(/\s+$/, "").slice(-1);
  const expectingPropertyName = /Expected double-quoted property name/.test(message)
    || /Expecting property name/.test(message);
  const expectingSeparator = /Expected ',' or '/.test(message);
  const expectingColon = /Expected ':' after property name/.test(message);
  // 候选文本必须比原来「更好」才接受：要么能整体解析，要么报错位置更靠后。
  // 模型常常一处输出同时犯 2~3 个错，所以不能要求一次就修到能解析。
  const parseErrorPosition = (value: string): number => {
    try {
      JSON.parse(value);
      return -2;
    } catch (parseError) {
      const m = /at position (\d+)/.exec(parseError instanceof Error ? parseError.message : String(parseError));
      return m ? Number(m[1]) : -1;
    }
  };
  const improves = (candidate: string): boolean => {
    const next = parseErrorPosition(candidate);
    if (next === -2) {
      return true;
    }
    if (next < 0) {
      return false;
    }
    return next > position;
  };
  // 从 index 处开始吃掉一段垃圾（空白 + 指定字符），保留后面的正经内容。
  // 模型常写成 `…,   - ,"key"`，光删掉 "-" 会留下两个逗号，所以要连着后面的逗号一起吃。
  // 注意 charClass 必须精确：像 `0.92}]` 这种，只该吃掉 }，不能把数组的 ] 一起吃掉。
  const dropCharRun = (value: string, index: number, charClass: RegExp): string => {
    const head = value.slice(0, index).replace(/\s+$/, "");
    let cursor = index;
    while (cursor < value.length && /\s/.test(value[cursor])) {
      cursor += 1;
    }
    while (cursor < value.length && charClass.test(value[cursor])) {
      cursor += 1;
      while (cursor < value.length && /\s/.test(value[cursor])) {
        cursor += 1;
      }
    }
    return head + value.slice(cursor);
  };
  // 只删掉 index 处的**一个**字符（连同它前面的空白）。
  // 括号必须一个一个删：成串删会把 "]}]}" 这种合法的收尾序列也吃掉。
  const dropOne = (value: string, index: number): string => {
    const head = value.slice(0, index).replace(/\s+$/, "");
    let cursor = index;
    while (cursor < value.length && /\s/.test(value[cursor])) {
      cursor += 1;
    }
    return head + value.slice(cursor + 1);
  };
  // 取 trimmedBefore 结尾的**最后一个完整字符串 token**（含首尾引号）。
  // 注意：不能直接找最后一个引号 —— 那是收尾引号，会把 token 切掉一半。
  const trailingQuoted = (): { text: string; length: number; prev: string } | null => {
    let inString = false;
    let escaped = false;
    let tokenStart = -1;
    for (let i = 0; i < trimmedBefore.length; i += 1) {
      const ch = trimmedBefore[i];
      if (inString) {
        if (escaped) {
          escaped = false;
          continue;
        }
        if (ch === "\\") {
          escaped = true;
          continue;
        }
        if (ch === "\"") {
          inString = false;
        }
        continue;
      }
      if (ch === "\"") {
        inString = true;
        tokenStart = i;
      }
    }
    if (inString || tokenStart < 0) {
      return null;
    }
    return {
      text: trimmedBefore.slice(tokenStart),
      length: trimmedBefore.length - tokenStart,
      prev: trimmedBefore.slice(0, tokenStart).replace(/\s+$/, "").slice(-1),
    };
  };
  const quotedHead = (value: string): { token: string; rest: string } | null => {
    if (!value.startsWith("\"")) {
      return null;
    }
    const end = findJsonStringEnd(value, 0);
    if (end < 0) {
      return null;
    }
    return { token: value.slice(0, end + 1), rest: skipWs(value.slice(end + 1)) };
  };
  // 扫出整段文本里「还没闭合」的括号栈（字符串内的括号不计）
  const openBracketStack = (value: string): string[] => {
    const stack: string[] = [];
    let inString = false;
    let escaped = false;
    for (let i = 0; i < value.length; i += 1) {
      const ch = value[i];
      if (inString) {
        if (escaped) {
          escaped = false;
          continue;
        }
        if (ch === "\\") {
          escaped = true;
          continue;
        }
        if (ch === "\"") {
          inString = false;
        }
        continue;
      }
      if (ch === "\"") {
        inString = true;
        continue;
      }
      if (ch === "{" || ch === "[") {
        stack.push(ch);
        continue;
      }
      // 必须按类型配对：模型在数组里写成对象的 } 时不能把 [ 弹掉，
      // 否则算出来的「还差几个收尾括号」是错的。
      if (ch === "}") {
        if (stack.length > 0 && stack[stack.length - 1] === "{") {
          stack.pop();
        }
        continue;
      }
      if (ch === "]") {
        if (stack.length > 0 && stack[stack.length - 1] === "[") {
          stack.pop();
        }
      }
    }
    return stack;
  };

  // ---------------------------------------------------------------- schema
  // 下面这一组工具用来「按 zod schema 反推被模型吞掉的 key 名」。
  // zod v4 的结构：def.type 是类型名；对象在 def.shape（对象或函数），
  // 数组在 def.element，optional/default 在 def.innerType，
  // z.preprocess 造出来的 pipe 是 def.in(变换) + def.out(真正的 schema)。

  // --- 规则 1：缺冒号 / 多余的 key ----------------------------------------
  if (expectingColon && nextChar === ",") {
    const afterComma = skipWs(after.slice(1));
    const quoted = quotedHead(afterComma);
    if (quoted) {
      if (quoted.rest.startsWith(":")) {
        // 2) 多余的 key："kind":"present","mention","summary":"…"
        //    前面那个带引号的 token 是多余的，整段删掉：
        //    → "kind":"present","summary":"…"
        const stray = trailingQuoted();
        if (stray && (stray.prev === "," || stray.prev === "{")) {
          // 连后面那个逗号一起删（不然会留下两个逗号）
          return trimmedBefore.slice(0, trimmedBefore.length - stray.length) + after.slice(1);
        }
      } else {
        // 1) 缺冒号："currentFact","沈砚…" → "currentFact":"沈砚…"
        return trimmedBefore + ":" + after.slice(1);
      }
    }
    if (afterComma && (afterComma[0] === "{" || afterComma[0] === "[" || /[0-9-]/.test(afterComma[0]))) {
      // 1b) 缺冒号且值不是字符串："a",{…} / "a",[…] / "a",123
      return trimmedBefore + ":" + after.slice(1);
    }
  }

  // --- 规则 2：key 缺左引号（mustAdvance": ）------------------------------
  if (expectingPropertyName) {
    const identifier = /^[A-Za-z_$][A-Za-z0-9_$]*/.exec(after);
    if (identifier) {
      const raw = identifier[0];
      const probe = position + raw.length;
      if (text[probe] === "\"") {
        const afterQuote = skipWs(text.slice(probe + 1));
        if (afterQuote.startsWith(":")) {
          return text.slice(0, position) + "\"" + raw + "\"" + text.slice(probe + 1);
        }
      } else if (text[probe] === ":" && (prevChar === "," || prevChar === "{")) {
        return text.slice(0, position) + "\"" + raw + "\"" + text.slice(probe);
      }
    }
  }

  // --- 规则 3：小数点被写成逗号（"verifiedCount":0,375}）------------------
  if (expectingPropertyName) {
    for (const commaPos of [position - 1, position]) {
      if (commaPos < 0 || text[commaPos] !== ",") {
        continue;
      }
      const head = text.slice(0, commaPos).replace(/\s+$/, "");
      if (!/:\s*-?\d+$/.test(head)) {
        continue;
      }
      const tail = skipWs(text.slice(commaPos + 1));
      if (/^\d{3}(?!\d)/.test(tail)) {
        return head + "." + tail;
      }
    }
  }

  // --- 规则 4：杂散标点（- * + | 和重复的逗号）---------------------------
  if (expectingPropertyName) {
    if (nextChar === "-" || nextChar === "*" || nextChar === "+" || nextChar === "|") {
      // 连后面那个多余的逗号一起吃掉（否则会留下 `,,"key"`）
      return dropCharRun(text, position, /[-*+|,]/);
    }
    // 重复的逗号：紧挨着当前逗号的前一个字符也是逗号（lastChar 是 position 前面的字符）
    if (nextChar === "," && (lastChar === "," || lastChar === "{")) {
      return dropCharRun(text, position, /[,]/);
    }
  }

  // --- 规则 5b：收尾多余的逗号（…, }  →  …}）---------------------------
  // 实测样本：…"targetEndChapterOrder":181]},}
  // 模型收尾时多打了一个逗号：逗号后面直接跟 } 就是多余的，删掉逗号即可。
  // 注意必须排在规则 5 前面 —— 规则 5 会把这个 } 删掉，
  // 然后规则 8 又会补一个 } 回来，形成「删了又补」的来回震荡。
  if (expectingPropertyName && nextChar === "}" && lastChar === ",") {
    const candidate = trimmedBefore.slice(0, -1) + after;
    if (improves(candidate)) {
      return candidate;
    }
  }

  // --- 规则 5：杂散括号 ---------------------------------------------------
  if (expectingPropertyName && (nextChar === "]" || nextChar === "}")) {
    return dropOne(text, position);
  }
  if (expectingSeparator && nextChar === "}") {
    // 模型在数组里把数组的收尾写成了对象的 } → 删掉它，让数组正常闭合。
    // （不能改成 ]：那样会多出一个 ]，把后面的结构顶歪。）
    return dropOne(text, position);
  }
  if (expectingSeparator && nextChar === "]") {
    return dropOne(text, position);
  }

  // --- 规则 6：数组里写成了对象（["…","confidence":0.95]）----------------
  if (expectingSeparator && nextChar === ":") {
    const stray = trailingQuoted();
    if (stray) {
      return trimmedBefore.slice(0, trimmedBefore.length - stray.length) + after.slice(1);
    }
  }

  // --- 规则 7：key 被转义（\"characterResourceDeltas"）--------------------
  // 两种形态：
  //   A) 开引号和收尾引号都被转义：\"key\"
  //   B) 只有开引号被转义、收尾是普通引号：\"key"    ← 实测样本
  //      …}]} ,\"characterResourceDeltas":[],\"payoffDeltas":[{…
  //      形态 B 里，成对的 \" 在很后面（下一个 key 的开引号），按「成对」找收尾会失败。
  if (expectingPropertyName && text[position] === "\\" && text[position + 1] === "\"") {
    let out = text;
    let cursor = position;
    for (let guard = 0; guard < 64; guard += 1) {
      const escaped = out.indexOf("\\\"", cursor);
      if (escaped < 0) {
        break;
      }
      const head = out.slice(0, escaped).replace(/\s+$/, "");
      const headLast = head.slice(-1);
      if (headLast !== "{" && headLast !== "," && headLast !== "}" && headLast !== "]") {
        break;
      }
      const bodyStart = escaped + 2;
      // 形态 A：收尾也是转义的 \"
      const paired = out.indexOf("\\\"", bodyStart);
      if (paired >= 0) {
        const pairedBody = out.slice(bodyStart, paired);
        if (!pairedBody.includes("\"")) {
          out = out.slice(0, escaped) + "\"" + pairedBody + "\"" + out.slice(paired + 2);
          cursor = escaped + pairedBody.length + 2;
          continue;
        }
      }
      // 形态 B：收尾是普通引号，后面必须紧跟 :
      let found = -1;
      let probe = bodyStart;
      while (probe < out.length) {
        const ch = out[probe];
        if (ch === "\\") {
          probe += 2;
          continue;
        }
        if (ch === "\"") {
          found = probe;
          break;
        }
        probe += 1;
      }
      if (found < 0) {
        break;
      }
      if (!skipWs(out.slice(found + 1)).startsWith(":")) {
        break;
      }
      const body = out.slice(bodyStart, found);
      if (body.includes("\"")) {
        break;
      }
      out = out.slice(0, escaped) + "\"" + body + "\"" + out.slice(found + 1);
      cursor = escaped + body.length + 2;
    }
    if (out !== text && improves(out)) {
      return out;
    }
  }

  // --- 规则 8：末尾漏了收尾括号 -------------------------------------------
  // 实测样本（artifact_delta 抽取）：
  //   …"characterStates":[{…},{…},"confidence":0.91,"requiresFullReconcile":false}
  // 模型把数组的 ] 漏掉了，解析器一路读到文本末尾还在数组里
  // → "Expected ',' or ']' after array element in JSON at position <末尾>"
  if (position >= text.length - 4
    && (/Expected ',' or '/.test(message)
      || /Unexpected end of JSON input/.test(message)
      || (expectingPropertyName && position >= text.length - 1))) {
    const stack = openBracketStack(text);
    if (stack.length > 0) {
      const closers = stack.slice().reverse().map((ch) => (ch === "{" ? "}" : "]")).join("");
      return text + closers;
    }
  }

  // --- 规则 9：孤儿值（该有 key 的位置却出现了值）------------------------
  // 形态 9a：多余的 "[" —— 实测样本 …"reveals":[…],["riskNotes":[…],…
  //   模型多打了一个 [，删掉就还原成合法结构，数据一条不丢。
  // 形态 9b：丢了 key —— 实测样本 …"stateDeltas":{…}},[{"resourceName":…},…]
  //   模型把 "characterResourceDeltas" 这个 key 整个吞了，只剩下数组值。
  //   这里用 schema 按元素字段签名反推 key 名。
  if (expectingPropertyName && (nextChar === "[" || nextChar === "{")) {
    if (nextChar === "[") {
      const afterBracket = skipWs(after.slice(1));
      const quoted = quotedHead(afterBracket);
      if (quoted && quoted.rest.startsWith(":")) {
        const candidate = text.slice(0, position) + text.slice(position + 1);
        if (improves(candidate)) {
          return candidate;
        }
      }
    }
    const inferred = inferMissingKeyFromSchema(__aiNovelStructuredSchemaHint, text, position);
    if (inferred) {
      const candidate = text.slice(0, position) + "\"" + inferred + "\":" + text.slice(position);
      if (improves(candidate)) {
        return candidate;
      }
    }
  }
  return null;
}

function applyJsonPositionRepairRound(text: string, error: unknown): string | null {
  const message = error instanceof Error ? error.message : String(error ?? "");
  const matched = /at position (\d+)/.exec(message);
  if (!matched) {
    return null;
  }
  const position = Number(matched[1]);
  // REPAIR_STRUCTURAL_SHAPE：先跑新的一族定点修补（缺冒号 / 缺引号 / 杂散标点 /
  // 小数点变逗号 / 数组里写成对象 / 孤儿值）。老的两条规则认不全这些形态，放前面先处理。
  const structural = applyJsonStructuralFixRound(text, error);
  if (structural && structural !== text) {
    return structural;
  }
  // REPAIR_STRAY_BRACKET（2026-11-12）：模型多吐了一个收尾括号。
  // 实测样本：..."targetEndChapterOrder":98}]}],  ← 比正确结构多了一个 ]
  // 解析器在对象内部要求 , 或 }，却遇到 ] / } → 说明这个收尾符多余，删掉即可。
  // 只动「精确报错位置」，不做偏移猜测，避免误删正常字符。
  if (position >= 0 && position < text.length) {
    const strayChar = text[position];
    // 只有「数组深度为 0」时这个 ] 才是多余的；
    // 数组深度 > 0 说明更可能是少了一个 }，删 ] 会把文本改得更糟。
    const strayBracket = strayChar === "]"
      && /Expected ',' or '\}'/.test(message)
      && jsonArrayDepthAt(text, position) <= 0;
    if (strayBracket) {
      const patched = text.slice(0, position) + text.slice(position + 1);
      try {
        JSON.parse(patched);
      } catch {
        // 整段还有别的问题：照样接受这次删除，让下一轮继续修
      }
      return patched;
    }
  }
  const inString = markJsonStringPositions(text);
  const offsets = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6];
  // 先按"字符串边界"判断找一遍；边界判断本身可能因未转义引号而错位，
  // 所以再跑一遍不看边界的候选。两轮都要求补丁后能整体解析。
  for (const trustBoundaries of [true, false]) {
    for (const offset of offsets) {
      let index = position + offset;
      if (index < 0 || index >= text.length) {
        continue;
      }
      while (index < text.length && /\s/.test(text[index])) {
        index += 1;
      }
      if (index >= text.length) {
        continue;
      }
      if (trustBoundaries && inString[index]) {
        continue;
      }
      const before = text.slice(0, index).replace(/\s+$/, "");
      if (!before) {
        continue;
      }
      const last = before[before.length - 1];
      const ch = text[index];
      // 情况一：key 没加引号 —— {a: 1} → {"a": 1}
      // REPAIR_DOTTED_KEY：key 前面可能多一个点（实测 ".issueIds: [...]"），
      // 补引号前要把多余的点去掉。
      if ((last === "{" || last === ",") && /[A-Za-z_$.]/.test(ch)) {
        let end = index;
        while (end < text.length && /[A-Za-z0-9_$.]/.test(text[end])) {
          end += 1;
        }
        let probe = end;
        while (probe < text.length && /\s/.test(text[probe])) {
          probe += 1;
        }
        if (text[probe] === ":") {
          const rawKey = text.slice(index, end);
          const cleanKey = rawKey.replace(/^\.+/, "");
          if (!cleanKey) {
            continue;
          }
          return text.slice(0, index) + "\"" + cleanKey + "\"" + text.slice(end);
        }
      }
      // 情况二：漏逗号 —— {"a":1 "b":2} → {"a":1,"b":2}
      // REPAIR_STRUCTURAL_SHAPE：起始字符从只有 " 扩到 " { [
      //   （数组里 [{…}{…}] 这种漏逗号很常见）。
      // 接受条件也放宽：整段能解析 **或** 新的报错位置比原来更靠后 ——
      //   模型常常同时犯 2~3 个错，要求整段解析成功会让这一轮直接放弃。
      if ((ch === "\"" || ch === "{" || ch === "[")
        && /[}\]"0-9a-zA-Z]/.test(last) && last !== "," && last !== "{" && last !== "[") {
        if (ch === "\"" && findJsonStringEnd(text, index) < 0) {
          continue;
        }
        const patched = text.slice(0, index) + "," + text.slice(index);
        try {
          JSON.parse(patched);
          return patched;
        } catch (nextError) {
          const nextMatched = /at position (\d+)/.exec(nextError instanceof Error ? nextError.message : "");
          const nextPosition = nextMatched ? Number(nextMatched[1]) : -1;
          if (trustBoundaries && nextPosition > position) {
            // 报错位置前进了 → 这次补逗号是对的，剩下的错交给下一轮
            return patched;
          }
          continue;
        }
      }
    }
  }
  return null;
}

function tryRepairMalformedJsonStructure(source: string, parseError: unknown): unknown | null {
  let extracted: string;
  try {
    extracted = extractJSONValue(source);
  } catch {
    return null;
  }
  // 第一步：整篇扫描式修补，能一次改掉多处
  const scanCandidates = [
    extracted,
    escapeJsonStringControlChars(extracted),
    repairJsonPunctuation(extracted),
    repairJsonPunctuation(escapeJsonStringControlChars(extracted)),
  ];
  for (const candidate of scanCandidates) {
    if (!candidate) {
      continue;
    }
    try {
      return JSON.parse(candidate) as unknown;
    } catch {
      continue;
    }
  }
  // 第二步：按解析错误位置逐处外科式修补，可多轮
  let current = repairJsonPunctuation(escapeJsonStringControlChars(extracted));
  for (let round = 0; round < 12; round += 1) {
    try {
      return JSON.parse(current) as unknown;
    } catch (error) {
      const patched = applyJsonPositionRepairRound(current, error);
      if (!patched || patched === current) {
        break;
      }
      current = patched;
    }
  }
  return null;
}

/**
 * REPAIR_CN_QUOTE（2026-11-12）：修复「模型用中文引号充当 JSON 字符串定界符」。
 *
 * 典型畸形（实测 gemini-2.5-flash 的 artifact_delta 抽取）：
 *   "evidence":"……语气沉稳，“梁东家手里的暗账原本……”},{
 * ASCII 收尾引号缺失，中文右引号 ” 出现在结构字符 } 之前，
 * 于是该字符串永不闭合，后面的内容全被吞进字符串 → JSON.parse 报
 * "Expected ',' or '}' after property value in JSON at position N"。
 *
 * 规则（只动确定是定界符的地方，正文里的中文引号一律保留）：
 *   - 字符串内遇到 ” 且其后（跳过空白）是 , } ] : → 换成 ASCII " 并结束该字符串；
 *   - 字符串内遇到 “ 或后面不是结构字符的 ” → 原样保留；
 *   - 字符串外遇到 “ → 换成 ASCII " 并进入字符串（模型把开引号写成中文）。
 */

/**
 * REPAIR_STRAY_BRACKET（2026-11-12）：删掉模型多吐的收尾方括号。
 *
 * 为什么必须在切子串之前做：
 *   extractJSONValue 靠括号配平找 JSON 边界，多一个 ] 会让配平永远回不到 0，
 *   直接抛「未检测到完整 JSON 值。」→ 后面基于解析器位置的位置式修补拿不到文本。
 *
 * 判定不依赖解析器：扫描字符串外的括号，某个 ] 出现时若没有未闭合的数组，它就是多余的。
 *
 * 安全阀：扫描结束时字符串仍未闭合 → 引号本身有问题，括号统计不可信 → 原样返回。
 */
function repairStrayClosingBrackets(source: string): string {
  const text = String(source ?? "");
  if (!text.includes("]")) {
    return text;
  }
  let out = "";
  let inString = false;
  let escaped = false;
  let arrayDepth = 0;
  let changed = false;
  for (let index = 0; index < text.length; index += 1) {
    const ch = text[index];
    if (escaped) {
      escaped = false;
      out += ch;
      continue;
    }
    if (ch === "\\") {
      escaped = true;
      out += ch;
      continue;
    }
    if (ch === "\"") {
      inString = !inString;
      out += ch;
      continue;
    }
    if (inString) {
      out += ch;
      continue;
    }
    if (ch === "[") {
      arrayDepth += 1;
      out += ch;
      continue;
    }
    if (ch === "]") {
      if (arrayDepth <= 0) {
        changed = true;
        continue;
      }
      arrayDepth -= 1;
      out += ch;
      continue;
    }
    out += ch;
  }
  if (inString) {
    return text;
  }
  return changed ? out : text;
}

function repairChineseQuoteDelimiters(source: string): string {
  const text = String(source ?? "");
  if (!text.includes("“") && !text.includes("”")) {
    return text;
  }
  let out = "";
  let inString = false;
  let started = false;
  for (let index = 0; index < text.length; index += 1) {
    const ch = text[index];
    if (!started) {
      // 只从第一个 { 或 [ 开始判断字符串状态，避免正文前缀里的引号干扰
      if (ch === "{" || ch === "[") {
        started = true;
      } else {
        out += ch;
        continue;
      }
    }
    if (!inString) {
      if (ch === "\"") {
        inString = true;
        out += ch;
        continue;
      }
      if (ch === "“") {
        inString = true;
        out += "\"";
        continue;
      }
      out += ch;
      continue;
    }
    if (ch === "\\") {
      out += ch;
      if (index + 1 < text.length) {
        out += text[index + 1];
        index += 1;
      }
      continue;
    }
    if (ch === "\"") {
      inString = false;
      out += ch;
      continue;
    }
    if (ch === "”") {
      let peek = index + 1;
      while (peek < text.length && /\s/.test(text[peek])) {
        peek += 1;
      }
      const next = peek < text.length ? text[peek] : "";
      if (next === "," || next === "}" || next === "]" || next === ":") {
        // 保留正文里的中文右引号，再补一个 ASCII 收尾引号（正文一字不丢）
        out += "”\"";
        inString = false;
        continue;
      }
      out += ch;
      continue;
    }
    out += ch;
  }
  return out;
}

function tryParseStructuredJsonValue(source: string): { parsed: unknown } | { error: string } {
  try {
    return {
      parsed: JSON.parse(extractJSONValue(source)) as unknown,
    };
  } catch (error) {
    // 解析失败时把原始内容前 800 字写进日志：真实畸形文本不用再猜。
    try {
      const dump = String(source ?? "");
      const oneLine = dump.replace(/[\r\n]+/g, " ");
      console.warn("[structured.parse] 无法解析的原始内容(前 12000 字): " + oneLine.slice(0, 12000));
    } catch {
      // 日志失败不影响主流程
    }
    // REPAIR_CN_QUOTE：先修「中文引号当收尾引号」。
    let working = source;
    const quoteFixed = repairChineseQuoteDelimiters(working);
    if (quoteFixed !== working) {
      working = quoteFixed;
      try {
        return {
          parsed: JSON.parse(extractJSONValue(working)) as unknown,
        };
      } catch {
        // 修完仍不合法，交给后续修补流程
      }
    }
    // REPAIR_STRAY_BRACKET：再删掉「多吐的收尾方括号」。
    // 必须放在切子串之前：多余的 ] 会让 extractJSONValue 的括号配平算不出来，
    // 后面的修复流水线会直接拿不到文本。
    const bracketFixed = repairStrayClosingBrackets(working);
    if (bracketFixed !== working) {
      working = bracketFixed;
      try {
        return {
          parsed: JSON.parse(extractJSONValue(working)) as unknown,
        };
      } catch {
        // 继续往下修
      }
    }
    const repaired = tryRepairMalformedJsonStructure(working, error);
    if (repaired !== null && repaired !== undefined) {
      return { parsed: repaired };
    }
    const fixed = tryFixTruncatedJson(working);
    if (fixed === working) {
      return {
        error: [
          "JSON 解析失败：",
          error instanceof Error ? error.message : String(error),
        ].join("\n"),
      };
    }
    try {
      return {
        parsed: JSON.parse(extractJSONValue(fixed)) as unknown,
      };
    } catch (fixedError) {
      return {
        error: [
          "JSON 解析失败：",
          error instanceof Error ? error.message : String(error),
          "截断修复后仍失败：",
          fixedError instanceof Error ? fixedError.message : String(fixedError),
        ].join("\n"),
      };
    }
  }
}

function tryUnwrapSingletonArrayWrapper<T>(
  parsed: unknown,
  schema: ZodType<T>,
): { data: T } | null {
  if (!Array.isArray(parsed) || parsed.length !== 1 || schemaAllowsTopLevelArray(schema)) {
    return null;
  }

  const unwrapped = schema.safeParse(parsed[0]);
  if (!unwrapped.success) {
    return null;
  }

  return {
    data: unwrapped.data,
  };
}

function formatZodErrors(error: ZodError): string {
  return error.issues
    .map((issue) => {
      const path = issue.path.length ? issue.path.join(".") : "(root)";
      return `- ${path}: ${issue.message}`;
    })
    .join("\n");
}

function formatIssuePath(path: Array<string | number>): string {
  return path.length > 0 ? path.join(".") : "(root)";
}

function normalizeIssuePath(path: readonly PropertyKey[]): Array<string | number> {
  return path.flatMap((segment) => {
    if (typeof segment === "string" || typeof segment === "number") {
      return [segment];
    }
    return [];
  });
}

function cloneJsonValue<T>(value: T): T {
  return value === undefined ? value : JSON.parse(JSON.stringify(value)) as T;
}

function getValueAtPath(root: unknown, path: Array<string | number>): unknown {
  let current = root;
  for (const segment of path) {
    if (Array.isArray(current) && typeof segment === "number") {
      current = current[segment];
      continue;
    }
    if (current && typeof current === "object" && !Array.isArray(current)) {
      current = (current as Record<string, unknown>)[String(segment)];
      continue;
    }
    return undefined;
  }
  return current;
}

function setValueAtPath(root: unknown, path: Array<string | number>, nextValue: unknown): unknown {
  if (path.length === 0) {
    return nextValue;
  }

  const parentPath = path.slice(0, -1);
  const leaf = path[path.length - 1]!;
  const parent = getValueAtPath(root, parentPath);
  if (Array.isArray(parent) && typeof leaf === "number") {
    parent[leaf] = nextValue;
  } else if (parent && typeof parent === "object" && !Array.isArray(parent)) {
    (parent as Record<string, unknown>)[String(leaf)] = nextValue;
  }
  return root;
}

function normalizeOversizedArrays<T>(
  parsed: unknown,
  error: ZodError,
  schema: ZodType<T>,
): { data: T; trimmedPaths: string[] } | null {
  let normalized = cloneJsonValue(parsed);
  const trimmedPaths: string[] = [];

  for (const issue of error.issues) {
    if (issue.code !== "too_big" || !issue.message.toLowerCase().includes("array")) {
      continue;
    }
    const maximum = typeof (issue as { maximum?: unknown }).maximum === "number"
      ? (issue as { maximum: number }).maximum
      : null;
    if (!Number.isInteger(maximum) || maximum === null || maximum < 0) {
      continue;
    }

    const issuePath = normalizeIssuePath(issue.path);
    const currentValue = getValueAtPath(normalized, issuePath);
    if (!Array.isArray(currentValue) || currentValue.length <= maximum) {
      continue;
    }

    normalized = setValueAtPath(normalized, issuePath, currentValue.slice(0, maximum));
    trimmedPaths.push(formatIssuePath(issuePath));
  }

  if (trimmedPaths.length === 0) {
    return null;
  }

  const final = schema.safeParse(normalized);
  if (!final.success) {
    return null;
  }

  return {
    data: final.data,
    trimmedPaths,
  };
}

function buildDiagnostics(input: {
  strategy: StructuredOutputStrategy;
  profile: StructuredOutputProfile;
  reasoningForcedOff?: boolean;
  fallbackAvailable?: boolean;
  fallbackUsed?: boolean;
  errorCategory?: StructuredOutputErrorCategory | null;
  response?: StructuredResponseEvidence;
}): StructuredOutputDiagnostics {
  return {
    response: input.response,
    strategy: input.strategy,
    profile: input.profile,
    reasoningForcedOff: input.reasoningForcedOff ?? false,
    fallbackAvailable: input.fallbackAvailable ?? false,
    fallbackUsed: input.fallbackUsed ?? false,
    errorCategory: input.errorCategory ?? null,
  };
}

export function logStructuredInvokeEvent(input: {
  event: string;
  label: string;
  provider?: LLMProvider;
  model?: string;
  taskType?: TaskType;
  latencyMs?: number;
  rawChars?: number;
  repairAttempt?: number;
  strategy?: StructuredOutputStrategy;
  errorCategory?: StructuredOutputErrorCategory | null;
  fallbackUsed?: boolean;
  reasoningForcedOff?: boolean;
}): void {
  console.info(
    [
      "[structured.invoke]",
      `event=${input.event}`,
      `label=${input.label}`,
      `provider=${input.provider ?? "default"}`,
      `model=${input.model ?? "default"}`,
      `taskType=${input.taskType ?? "planner"}`,
      input.strategy ? `strategy=${input.strategy}` : "",
      input.errorCategory ? `errorCategory=${input.errorCategory}` : "",
      typeof input.repairAttempt === "number" ? `repairAttempt=${input.repairAttempt}` : "",
      typeof input.latencyMs === "number" ? `latencyMs=${input.latencyMs}` : "",
      typeof input.rawChars === "number" ? `rawChars=${input.rawChars}` : "",
      input.fallbackUsed ? "fallbackUsed=true" : "",
      input.reasoningForcedOff ? "reasoningForcedOff=true" : "",
    ].filter(Boolean).join(" "),
  );
}

export function buildStructuredError(input: {
  message: string;
  category: StructuredOutputErrorCategory;
  strategy: StructuredOutputStrategy;
  profile: StructuredOutputProfile;
  retryWithNextStrategy?: boolean;
  reasoningForcedOff?: boolean;
  fallbackAvailable?: boolean;
  fallbackUsed?: boolean;
  response?: StructuredResponseEvidence;
  cause?: unknown;
}): StructuredOutputError {
  return new StructuredOutputError({
    message: input.message,
    category: input.category,
    retryWithNextStrategy: input.retryWithNextStrategy,
    cause: input.cause,
    diagnostics: buildDiagnostics({
      strategy: input.strategy,
      profile: input.profile,
      reasoningForcedOff: input.reasoningForcedOff,
      fallbackAvailable: input.fallbackAvailable,
      fallbackUsed: input.fallbackUsed,
      errorCategory: input.category,
      response: input.response,
    }),
  });
}

export function wrapStructuredInvokeError(input: {
  label: string;
  error: unknown;
  strategy: StructuredOutputStrategy;
  profile: StructuredOutputProfile;
  rawContent?: string;
  reasoningForcedOff?: boolean;
  fallbackAvailable?: boolean;
  fallbackUsed?: boolean;
}): StructuredOutputError {
  if (input.error instanceof StructuredOutputError) {
    return input.error;
  }
  const category = classifyStructuredOutputFailure({
    error: input.error,
    rawContent: input.rawContent,
  });
  const message = input.error instanceof Error
    ? input.error.message
    : typeof input.error === "string"
      ? input.error
      : `[${input.label}] Structured output failed.`;
  return buildStructuredError({
    message,
    cause: input.error,
    category,
    strategy: input.strategy,
    profile: input.profile,
    reasoningForcedOff: input.reasoningForcedOff,
    fallbackAvailable: input.fallbackAvailable,
    fallbackUsed: input.fallbackUsed,
  });
}

function getRepairHelpers<T>(schema: ZodType<T>) {
  return {
    // REPAIR_STRUCTURAL_SHAPE：把 schema 挂上去，解析时才能反推被吞掉的 key
    tryParseStructuredJsonValue: (source: string) =>
      withSchemaHint(schema, () => tryParseStructuredJsonValue(source)),
    tryUnwrapSingletonArrayWrapper,
    normalizeOversizedArrays,
    formatZodErrors,
    logStructuredInvokeEvent,
  };
}

export function shouldUseJsonObjectResponseFormat<T>(
  provider: LLMProvider,
  model: string | undefined,
  schema: ZodType<T>,
  baseURL?: string,
): boolean {
  const profile = resolveStructuredOutputProfile({
    provider,
    model,
    baseURL,
    executionMode: "structured",
  });
  return selectStructuredOutputStrategy(profile, schema) === "json_object";
}

export async function parseStructuredLlmRawContentDetailed<T>(
  input: StructuredInvokeRawParseInput<T>,
): Promise<StructuredInvokeResult<T>> {
  const runtimeSchema: ZodType<T> = relaxGeneratedContentSchema(input.schema);
  const diagnostics = buildDiagnostics({
    strategy: input.strategy,
    profile: input.profile,
    reasoningForcedOff: input.reasoningForcedOff,
    fallbackAvailable: input.fallbackAvailable,
    fallbackUsed: input.fallbackUsed,
  });
  const responseFailureReason = input.responseEvidence
    ? classifyResponseEvidence(input.responseEvidence)
    : input.rawContent.trim() ? null : "empty_response";
  if (input.responseEvidence) input.responseEvidence.failureReason = responseFailureReason;
  if (terminalResponseReason(responseFailureReason)) {
    const failure = describeStructuredResponseFailure(responseFailureReason!);
    throw buildStructuredError({
      message: `[${input.label}] ${failure.message}`,
      category: failure.category,
      retryWithNextStrategy: responseFailureReason === "empty_response" && input.strategy !== "prompt_json",
      strategy: input.strategy,
      profile: input.profile,
      reasoningForcedOff: input.reasoningForcedOff,
      fallbackAvailable: input.fallbackAvailable,
      fallbackUsed: input.fallbackUsed,
      response: input.responseEvidence,
    });
  }
  // REPAIR_STRUCTURAL_SHAPE：解析期间挂上 schema（供反推丢失的 key 用）
  const initialParse = withSchemaHint(runtimeSchema, () => tryParseStructuredJsonValue(input.rawContent));
  const parseErrorMessage = "error" in initialParse ? initialParse.error : "";
  const parsed = "parsed" in initialParse ? initialParse.parsed : null;

  const maxRepairAttempts = input.maxRepairAttempts ?? 1;
  if (parseErrorMessage) {
    await recordInvocationParseFailure(parseErrorMessage);
    for (let attempt = 1; attempt <= maxRepairAttempts; attempt += 1) {
      try {
        return {
          data: await repairWithLlm<T>({
            ...input,
            schema: runtimeSchema,
          }, input.rawContent, parseErrorMessage, attempt, getRepairHelpers<T>(runtimeSchema)),
          repairUsed: true,
          repairAttempts: attempt,
          diagnostics,
          tokenUsage: input.tokenUsage ?? null,
        };
      } catch (repairError) {
        if (repairError instanceof StructuredOutputError && terminalResponseReason(repairError.diagnostics.response?.failureReason)) throw repairError;
        if (attempt >= maxRepairAttempts) {
          throw buildStructuredError({
            message: `[${input.label}] JSON 解析失败且修复未成功。错误：${repairError instanceof Error ? repairError.message : String(repairError)}`,
            cause: repairError,
            category: classifyStructuredOutputFailure({
              error: repairError,
              rawContent: input.rawContent,
            }),
            strategy: input.strategy,
            profile: input.profile,
            reasoningForcedOff: input.reasoningForcedOff,
            fallbackAvailable: input.fallbackAvailable,
            fallbackUsed: input.fallbackUsed,
          });
        }
      }
    }
    throw buildStructuredError({
      message: `[${input.label}] JSON 解析未通过。`,
      category: classifyStructuredOutputFailure({ error: parseErrorMessage }) === "incomplete_json" ? "incomplete_json" : "malformed_json",
      strategy: input.strategy, profile: input.profile,
      reasoningForcedOff: input.reasoningForcedOff, fallbackAvailable: input.fallbackAvailable, fallbackUsed: input.fallbackUsed,
    });
  }

  const first = runtimeSchema.safeParse(parsed);
  if (first.success) {
    return {
      data: first.data,
      repairUsed: false,
      repairAttempts: 0,
      diagnostics,
      tokenUsage: input.tokenUsage ?? null,
    };
  }

  const unwrappedInitial = tryUnwrapSingletonArrayWrapper(parsed, runtimeSchema);
  if (unwrappedInitial) {
    logStructuredInvokeEvent({
      event: "unwrapped_singleton_array",
      label: input.label,
      provider: input.provider,
      model: input.model,
      taskType: input.taskType,
      strategy: input.strategy,
      fallbackUsed: input.fallbackUsed,
      reasoningForcedOff: input.reasoningForcedOff,
    });
    return {
      data: unwrappedInitial.data,
      repairUsed: false,
      repairAttempts: 0,
      diagnostics,
      tokenUsage: input.tokenUsage ?? null,
    };
  }

  const normalizedInitial = normalizeOversizedArrays(parsed, first.error, runtimeSchema);
  if (normalizedInitial) {
    logStructuredInvokeEvent({
      event: "normalized",
      label: input.label,
      provider: input.provider,
      model: input.model,
      taskType: input.taskType,
      strategy: input.strategy,
      fallbackUsed: input.fallbackUsed,
      reasoningForcedOff: input.reasoningForcedOff,
    });
    return {
      data: normalizedInitial.data,
      repairUsed: false,
      repairAttempts: 0,
      diagnostics,
      tokenUsage: input.tokenUsage ?? null,
    };
  }

  let zodError: ZodError = first.error;
  await recordInvocationParseFailure(formatZodErrors(zodError));
  for (let attempt = 1; attempt <= maxRepairAttempts; attempt += 1) {
    try {
      return {
        data: await repairWithLlm<T>({
          ...input,
          schema: runtimeSchema,
        }, input.rawContent, `Zod 校验错误：\n${formatZodErrors(zodError)}`, attempt, getRepairHelpers<T>(runtimeSchema)),
        repairUsed: true,
        repairAttempts: attempt,
        diagnostics,
        tokenUsage: input.tokenUsage ?? null,
      };
    } catch (error) {
      if (error instanceof StructuredOutputError && terminalResponseReason(error.diagnostics.response?.failureReason)) throw error;
      if (attempt >= maxRepairAttempts) {
        throw buildStructuredError({
          message: `[${input.label}] LLM 输出经修复后仍未通过 Schema 校验。错误：${error instanceof Error ? error.message : String(error)}`,
          category: "schema_mismatch",
          cause: error,
          strategy: input.strategy,
          profile: input.profile,
          reasoningForcedOff: input.reasoningForcedOff,
          fallbackAvailable: input.fallbackAvailable,
          fallbackUsed: input.fallbackUsed,
        });
      }
      if (error instanceof z.ZodError) {
        zodError = error as ZodError;
      }
    }
  }

  throw buildStructuredError({
    message: `[${input.label}] LLM 输出经修复后仍未通过 Schema 校验。错误：${formatZodErrors(zodError)}`,
    category: "schema_mismatch",
    strategy: input.strategy,
    profile: input.profile,
    reasoningForcedOff: input.reasoningForcedOff,
    fallbackAvailable: input.fallbackAvailable,
    fallbackUsed: input.fallbackUsed,
  });
}
