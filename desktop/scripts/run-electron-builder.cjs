const fs = require("node:fs");
const { createRequire } = require("node:module");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { parseArgs } = require("node:util");

const repoRoot = path.resolve(__dirname, "..", "..");
const desktopDir = path.resolve(__dirname, "..");
const nsisEnvOverrideName = "AI_NOVEL_EB_NSIS_TEMPLATES_DIR";
const traversalEnvOverrideName = "AI_NOVEL_EB_FORCE_TRAVERSAL";
const nsisUtilOriginal = 'exports.nsisTemplatesDir = (0, pathManager_1.getTemplatePath)("nsis");';
const nsisUtilPatched =
  'exports.nsisTemplatesDir = process.env.AI_NOVEL_EB_NSIS_TEMPLATES_DIR && process.env.AI_NOVEL_EB_NSIS_TEMPLATES_DIR.trim() ? path.resolve(process.env.AI_NOVEL_EB_NSIS_TEMPLATES_DIR.trim()) : (0, pathManager_1.getTemplatePath)("nsis");';
const appFileCopierOriginal = 'const pmApproaches = [await packager.getPackageManager(), node_module_collector_1.PM.TRAVERSAL];';
const appFileCopierPatched =
  'const pmApproaches = process.env.AI_NOVEL_EB_FORCE_TRAVERSAL === "true" ? [node_module_collector_1.PM.TRAVERSAL] : [await packager.getPackageManager(), node_module_collector_1.PM.TRAVERSAL];';
// app-builder (Go 二进制) 解压 winCodeSign 时会带上 `-snld`，7-Zip 21.07 会尝试创建
// darwin/10.12/lib 下的符号链接（libssl.dylib / libcrypto.dylib）。Windows 未开开发者模式
// 且会话非管理员时 CreateSymbolicLink 失败 → 7za 退出码 2 → 整个打包被判失败
// （其余文件其实都已解出，只有这 2 个符号链接失败）。
// `-snl-` 会跳过符号链接处理，7za 返回 0，Windows 构建所需的
// rcedit-x64.exe / signtool.exe / appxAssets 均完好。两者同为 5 字符，可等长原地替换。
const SEVEN_ZIP_SYMLINK_FLAG_OLD = Buffer.from("-snld");
const SEVEN_ZIP_SYMLINK_FLAG_NEW = Buffer.from("-snl-");
const electronBuilderPackageJson = require.resolve("electron-builder/package.json", { paths: [desktopDir, repoRoot] });
const electronBuilderRequire = createRequire(electronBuilderPackageJson);

function resolveModule(request) {
  try {
    return require.resolve(request, { paths: [desktopDir, repoRoot] });
  } catch (error) {
    if (error && error.code === "MODULE_NOT_FOUND") {
      return electronBuilderRequire.resolve(request);
    }
    throw error;
  }
}

function patchFileInPlace(moduleRequest, originalSource, patchedSource, description) {
  const modulePath = resolveModule(moduleRequest);
  const targets = new Set([modulePath, fs.realpathSync(modulePath)]);

  for (const targetPath of targets) {
    const source = fs.readFileSync(targetPath, "utf8");

    if (source.includes(patchedSource)) {
      continue;
    }

    if (!source.includes(originalSource)) {
      throw new Error(`Unable to patch electron-builder ${description} at ${targetPath}. Expected marker was not found.`);
    }

    fs.writeFileSync(targetPath, source.replace(originalSource, patchedSource), "utf8");
  }
}

function ensurePatchedElectronBuilder(isWindowsTarget) {
  if (isWindowsTarget) patchFileInPlace(
    "app-builder-lib/out/targets/nsis/nsisUtil.js",
    nsisUtilOriginal,
    nsisUtilPatched,
    "NSIS util",
  );
  patchFileInPlace(
    "app-builder-lib/out/util/appFileCopier.js",
    appFileCopierOriginal,
    appFileCopierPatched,
    "app file copier",
  );
}

/**
 * 定位 app-builder (Go 二进制) 的可执行文件。
 * app-builder-bin 是 builder-util 的依赖（pnpm 严格模式下顶层不可见），
 * 先按包名解析，失败再直接扫 node_modules/.pnpm 目录。
 */
function locateAppBuilderExecutables() {
  const found = new Set();
  const tryResolve = (request, basePaths) => {
    try {
      return require.resolve(request, { paths: basePaths });
    } catch {
      return null;
    }
  };

  let packageJsonPath = tryResolve("app-builder-bin/package.json", [desktopDir, repoRoot]);
  if (!packageJsonPath) {
    const builderUtilPkg = tryResolve("builder-util/package.json", [desktopDir, repoRoot]);
    if (builderUtilPkg) {
      try {
        packageJsonPath = createRequire(builderUtilPkg).resolve("app-builder-bin/package.json");
      } catch {
        packageJsonPath = null;
      }
    }
  }
  if (packageJsonPath) {
    const packageRoot = path.dirname(packageJsonPath);
    for (const relative of ["win/x64/app-builder.exe", "win/ia32/app-builder.exe", "win/arm64/app-builder.exe"]) {
      const target = path.join(packageRoot, relative);
      if (fs.existsSync(target)) found.add(target);
    }
  }

  // 兜底：直接扫 pnpm store
  const pnpmDir = path.join(repoRoot, "node_modules", ".pnpm");
  if (fs.existsSync(pnpmDir)) {
    for (const entry of fs.readdirSync(pnpmDir)) {
      if (!entry.startsWith("app-builder-bin@")) continue;
      const packageRoot = path.join(pnpmDir, entry, "node_modules", "app-builder-bin");
      for (const relative of ["win/x64/app-builder.exe", "win/ia32/app-builder.exe", "win/arm64/app-builder.exe"]) {
        const target = path.join(packageRoot, relative);
        if (fs.existsSync(target)) found.add(target);
      }
    }
  }

  return [...found];
}

/**
 * 原地修补 app-builder 二进制里的 7za 符号链接开关（`-snld` → `-snl-`）。
 * 详见文件头常量处的说明。幂等：已替换过则直接跳过。
 */
function patchAppBuilderSymlinkFlag() {
  const targets = locateAppBuilderExecutables();
  if (!targets.length) {
    console.log("[dist:desktop] 未找到 app-builder 二进制，跳过 7za 符号链接开关修补");
    return;
  }
  const oldFlag = SEVEN_ZIP_SYMLINK_FLAG_OLD.toString("latin1");
  const newFlag = SEVEN_ZIP_SYMLINK_FLAG_NEW.toString("latin1");
  let patched = 0;
  for (const target of targets) {
    const data = fs.readFileSync(target);
    let count = 0;
    let index = data.indexOf(SEVEN_ZIP_SYMLINK_FLAG_OLD);
    while (index >= 0) {
      count += 1;
      index = data.indexOf(SEVEN_ZIP_SYMLINK_FLAG_OLD, index + 1);
    }
    if (!count) continue;
    const backup = `${target}.bak-snld`;
    if (!fs.existsSync(backup)) fs.copyFileSync(target, backup);
    const patchedBuffer = Buffer.from(data.toString("latin1").split(oldFlag).join(newFlag), "latin1");
    fs.writeFileSync(target, patchedBuffer);
    console.log(`[dist:desktop] 已修补 ${path.basename(path.dirname(target))}/app-builder.exe 的 7za 符号链接开关（${count} 处）`);
    patched += 1;
  }
  if (!patched) {
    console.log("[dist:desktop] 7za 符号链接开关无需修补");
  }
}

function resolveShortNsisTemplateDir() {
  const installerTemplate = resolveModule("app-builder-lib/templates/nsis/installer.nsi");
  const templatesDir = path.dirname(installerTemplate);
  const mirroredTemplatesRoot = path.join(desktopDir, "build", "electron-builder-templates");
  const mirroredNsisTemplatesDir = path.join(mirroredTemplatesRoot, "nsis");

  if (!fs.existsSync(templatesDir)) {
    throw new Error(`NSIS template directory was not found at ${templatesDir}.`);
  }

  fs.rmSync(mirroredNsisTemplatesDir, { recursive: true, force: true });
  fs.mkdirSync(mirroredTemplatesRoot, { recursive: true });
  fs.cpSync(templatesDir, mirroredNsisTemplatesDir, { recursive: true, force: true });

  return mirroredNsisTemplatesDir;
}

function firstNonEmpty(...values) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function resolvePublishMode(args) {
  const modes = parseArgs({ args, options: { publish: { type: "string", short: "p", multiple: true } }, strict: false, allowPositionals: true }).values.publish ?? [];
  if (modes.length > 1) throw new Error("Provide exactly one publish mode.");
  return modes[0];
}

function normalizeBuildEnvironment(sourceEnv, args) {
  const env = { ...sourceEnv };
  const releaseChannel = firstNonEmpty(env.AI_NOVEL_RELEASE_CHANNEL, "beta").toLowerCase();
  const publishMode = resolvePublishMode(args);
  const isPublishRequested = publishMode != null && publishMode !== "never";
  const isWindowsTarget = args.includes("--win") || (process.platform === "win32" && !args.includes("--mac") && !args.includes("--linux"));
  const allowUnsignedRelease =
    firstNonEmpty(
      env.AI_NOVEL_ALLOW_UNSIGNED_RELEASE,
      env.AI_NOVEL_ALLOW_UNSIGNED_WINDOWS_RELEASE,
    ).toLowerCase() === "true";

  const signingLink = firstNonEmpty(
    env.CSC_LINK,
    env.WIN_CSC_LINK,
    env.AI_NOVEL_WINDOWS_CSC_LINK,
    env.AI_NOVEL_WINDOWS_CSC_FILE,
  );
  const signingPassword = firstNonEmpty(
    env.CSC_KEY_PASSWORD,
    env.WIN_CSC_KEY_PASSWORD,
    env.AI_NOVEL_WINDOWS_CSC_KEY_PASSWORD,
    env.AI_NOVEL_WINDOWS_CSC_PASSWORD,
  );
  const githubToken = firstNonEmpty(env.GH_TOKEN, env.GITHUB_TOKEN, env.AI_NOVEL_GITHUB_TOKEN);

  if (signingLink) {
    env.CSC_LINK = signingLink;
  }
  if (signingPassword) {
    env.CSC_KEY_PASSWORD = signingPassword;
  }
  if (githubToken) {
    env.GH_TOKEN = githubToken;
  }

  const hasSigning = Boolean(signingLink);
  if (isWindowsTarget && !releaseChannel.startsWith("beta") && !hasSigning && !allowUnsignedRelease) {
    throw new Error(
      "Public Windows desktop releases require signing material. Provide CSC_LINK/WIN_CSC_LINK first, or explicitly allow an unsigned release.",
    );
  }

  if (isPublishRequested && !env.GH_TOKEN) {
    throw new Error("GitHub publish requested but no GH_TOKEN/GITHUB_TOKEN was provided.");
  }

  console.log(
    `[dist:desktop] releaseChannel=${releaseChannel} publish=${isPublishRequested ? "yes" : "no"} signing=${hasSigning ? "configured" : allowUnsignedRelease ? "unsigned-opt-in" : "unsigned-beta"}`,
  );

  return env;
}

function main() {
  const targetArgs = process.argv.slice(2);
  const forceTests = targetArgs.includes("--force-tests");
  const forwardedArgs = targetArgs.filter(arg => arg !== "--force-tests");
  const isWindowsTarget = forwardedArgs.includes("--win") || (process.platform === "win32" && !forwardedArgs.includes("--mac") && !forwardedArgs.includes("--linux"));
  const electronBuilderCli = resolveModule("electron-builder/cli.js");
  const args = ["--config", "electron-builder.config.cjs", ...forwardedArgs];
  if (resolvePublishMode(args) == null) args.push("--publish", "never");
  const env = normalizeBuildEnvironment(process.env, args);
  if (resolvePublishMode(args) !== "never") {
    require("../../scripts/desktop-release/guard.cjs").verifyUpload(repoRoot, { env, forceTests });
  }
  ensurePatchedElectronBuilder(isWindowsTarget);
  if (isWindowsTarget) patchAppBuilderSymlinkFlag();
  const shortNsisTemplatesDir = isWindowsTarget ? resolveShortNsisTemplateDir() : undefined;

  if (shortNsisTemplatesDir) console.log(`[dist:desktop] using NSIS templates from ${shortNsisTemplatesDir}`);

  execFileSync(process.execPath, [electronBuilderCli, ...args], {
    cwd: desktopDir,
    stdio: "inherit",
    env: {
      ...env,
      ...(shortNsisTemplatesDir ? { [nsisEnvOverrideName]: shortNsisTemplatesDir } : {}),
      [traversalEnvOverrideName]: "true",
    },
  });
}

module.exports = { normalizeBuildEnvironment, resolvePublishMode };
if (require.main === module) {
  try {
    main();
  } catch (error) {
    console.error("[dist:desktop] failed.", error);
    process.exit(1);
  }
}
