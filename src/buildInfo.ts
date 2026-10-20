import type { BuildInfo } from './core/metadata';

/** Replaced at build / dev-server start by vite.config.ts `define`. */
declare const __BUILD_INFO__: BuildInfo;

export const BUILD_INFO: BuildInfo = __BUILD_INFO__;
