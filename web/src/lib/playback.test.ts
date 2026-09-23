import { describe, expect, it } from 'vitest'
import type { RenderFrame } from '../api'
import { formatMs, holdTicks, onionNeighbours, pairOf, playbackOrder } from './playback'

// 3 pages with 2 in-betweens per pair: K i i K i i K
const frames: RenderFrame[] = [
  { index: 0, key: true, source: 0, t: 0, url: 'a' },
  { index: 1, key: false, source: 0, t: 1 / 3, url: 'b' },
  { index: 2, key: false, source: 0, t: 2 / 3, url: 'c' },
  { index: 3, key: true, source: 1, t: 0, url: 'd' },
  { index: 4, key: false, source: 1, t: 1 / 3, url: 'e' },
  { index: 5, key: false, source: 1, t: 2 / 3, url: 'f' },
  { index: 6, key: true, source: 2, t: 0, url: 'g' },
]

describe('playbackOrder', () => {
  it('plays everything in order by default', () => {
    expect(playbackOrder(frames, { keysOnly: false, pingpong: false })).toEqual([0, 1, 2, 3, 4, 5, 6])
  })
  it('can show only the original pages', () => {
    expect(playbackOrder(frames, { keysOnly: true, pingpong: false })).toEqual([0, 3, 6])
  })
  it('ping-pongs without repeating the end frames', () => {
    expect(playbackOrder(frames, { keysOnly: true, pingpong: true })).toEqual([0, 3, 6, 3])
  })
  it('handles tiny sequences', () => {
    expect(playbackOrder(frames.slice(0, 1), { keysOnly: false, pingpong: true })).toEqual([0])
  })
})

describe('holdTicks', () => {
  it('holds key frames for the length of their in-betweens when keys-only', () => {
    expect(holdTicks(frames, 0, true)).toBe(3)
    expect(holdTicks(frames, 6, true)).toBe(1)
    expect(holdTicks(frames, 0, false)).toBe(1)
  })
})

describe('onionNeighbours', () => {
  it('finds the surrounding pages', () => {
    expect(onionNeighbours(frames, 4)).toEqual({ prev: 3, next: 6 })
    expect(onionNeighbours(frames, 0)).toEqual({ prev: undefined, next: 3 })
  })
})

describe('misc', () => {
  it('maps frames to page pairs', () => {
    expect(pairOf(frames[4], 2)).toBe(1)
    expect(pairOf(frames[6], 2)).toBe(1)
  })
  it('formats durations', () => {
    expect(formatMs(3.14159)).toBe('3.1 ms')
    expect(formatMs(250.4)).toBe('250 ms')
    expect(formatMs(1500)).toBe('1.5 s')
    expect(formatMs(25000)).toBe('25 s')
  })
})
