// Pure helpers for the player; unit-tested in playback.test.ts.
import type { RenderFrame } from '../api'

export interface PlaybackOptions {
  /** Show only the original pages (what v1 looked like without interpolation). */
  keysOnly: boolean
  /** Play forward then backward instead of jumping back to the start. */
  pingpong: boolean
}

/** Indices (into `frames`) in the order they are shown for one loop. */
export function playbackOrder(frames: RenderFrame[], opts: PlaybackOptions): number[] {
  const base = frames.map((f, i) => ({ f, i })).filter(({ f }) => !opts.keysOnly || f.key)
  const order = base.map(({ i }) => i)
  if (opts.pingpong && order.length > 2) return [...order, ...order.slice(1, -1).reverse()]
  return order
}

/**
 * Keys-only playback should last as long as the full animation, so each page is
 * held for (inbetweens + 1) ticks. Returns how many ticks frame `index` is shown.
 */
export function holdTicks(frames: RenderFrame[], index: number, keysOnly: boolean): number {
  if (!keysOnly) return 1
  let n = 1
  for (let j = index + 1; j < frames.length && !frames[j].key; j++) n++
  return n
}

/** Frames whose ink is shown faintly behind the current one (previous/next pages). */
export function onionNeighbours(frames: RenderFrame[], index: number): { prev?: number; next?: number } {
  let prev: number | undefined
  let next: number | undefined
  for (let j = index - 1; j >= 0; j--)
    if (frames[j].key) {
      prev = j
      break
    }
  for (let j = index + 1; j < frames.length; j++)
    if (frames[j].key) {
      next = j
      break
    }
  return { prev, next }
}

/** Which page pair (flow visualisation) a frame belongs to. */
export function pairOf(frame: RenderFrame, pairCount: number): number {
  return Math.min(frame.source, Math.max(0, pairCount - 1))
}

export function formatMs(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(ms >= 10000 ? 0 : 1)} s`
  if (ms >= 10) return `${Math.round(ms)} ms`
  return `${ms.toFixed(1)} ms`
}
