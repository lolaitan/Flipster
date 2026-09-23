import clsx from 'clsx'
import { ChevronLeft, ChevronRight, Download, Pause, Play } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, type Render } from '../api'
import { holdTicks, onionNeighbours, pairOf, playbackOrder } from '../lib/playback'
import { buttonClass } from './buttonClass'
import { Button, Segmented, Toggle } from './ui'

const FPS = [6, 12, 24] as const

function useImages(urls: string[]) {
  const [progress, setProgress] = useState<{ urls: string[]; count: number }>({ urls, count: 0 })
  const images = useMemo(() => urls.map(() => new Image()), [urls])
  useEffect(() => {
    let alive = true
    images.forEach((img, i) => {
      img.decoding = 'async'
      img.src = urls[i]
      img
        .decode()
        .catch(() => undefined)
        .finally(() => {
          if (alive) setProgress((p) => (p.urls === urls ? { urls, count: p.count + 1 } : { urls, count: 1 }))
        })
    })
    return () => {
      alive = false
    }
  }, [images, urls])
  return { images, loaded: progress.urls === urls ? progress.count : 0 }
}

/** Ink turned into a colour (paper stays white), for onion-skin layers. */
function tinted(img: HTMLImageElement, color: string, cache: Map<HTMLImageElement, HTMLCanvasElement>) {
  let c = cache.get(img)
  if (c) return c
  c = document.createElement('canvas')
  c.width = img.naturalWidth
  c.height = img.naturalHeight
  const g = c.getContext('2d')!
  g.drawImage(img, 0, 0)
  g.globalCompositeOperation = 'screen'
  g.fillStyle = color
  g.fillRect(0, 0, c.width, c.height)
  cache.set(img, c)
  return c
}

export function Player({ render }: { render: Render }) {
  const frames = render.frames
  const frameUrls = useMemo(() => frames.map((f) => f.url), [frames])
  const { images, loaded } = useImages(frameUrls)
  const flowUrls = useMemo(() => render.flow_urls, [render.flow_urls])
  const { images: flowImages } = useImages(flowUrls)

  const [playing, setPlaying] = useState(true)
  const [fps, setFps] = useState<number>(12)
  const [keysOnly, setKeysOnly] = useState(false)
  const [pingpong, setPingpong] = useState(false)
  const [onion, setOnion] = useState(false)
  const [showFlow, setShowFlow] = useState(false)
  const [pos, setPos] = useState(0)
  const posRef = useRef(0)
  useEffect(() => {
    posRef.current = pos
  }, [pos])

  const order = useMemo(() => playbackOrder(frames, { keysOnly, pingpong }), [frames, keysOnly, pingpong])
  const current = order[Math.min(pos, order.length - 1)] ?? 0
  const canvas = useRef<HTMLCanvasElement>(null)
  const tintCache = useRef({ prev: new Map<HTMLImageElement, HTMLCanvasElement>(), next: new Map<HTMLImageElement, HTMLCanvasElement>() })
  const ready = loaded >= frames.length && frames.length > 0

  const restart = <T,>(set: (v: T) => void) => (v: T) => {
    set(v)
    setPos(0)
  }

  // draw
  useEffect(() => {
    const c = canvas.current
    const img = images[current]
    if (!c || !img || !img.complete || !img.naturalWidth) return
    if (c.width !== img.naturalWidth || c.height !== img.naturalHeight) {
      c.width = img.naturalWidth
      c.height = img.naturalHeight
    }
    const g = c.getContext('2d')!
    g.globalCompositeOperation = 'source-over'
    g.globalAlpha = 1
    g.drawImage(img, 0, 0)
    if (onion) {
      const { prev, next } = onionNeighbours(frames, current)
      g.globalCompositeOperation = 'multiply'
      g.globalAlpha = 0.35
      if (prev !== undefined && images[prev]?.naturalWidth) g.drawImage(tinted(images[prev], '#e0707a', tintCache.current.prev), 0, 0, c.width, c.height)
      if (next !== undefined && images[next]?.naturalWidth) g.drawImage(tinted(images[next], '#2f5bd3', tintCache.current.next), 0, 0, c.width, c.height)
    }
    if (showFlow && flowImages.length) {
      const f = flowImages[pairOf(frames[current], flowImages.length)]
      if (f?.naturalWidth) {
        g.globalCompositeOperation = 'multiply'
        g.globalAlpha = 0.85
        g.drawImage(f, 0, 0, c.width, c.height)
      }
    }
  }, [current, images, loaded, onion, showFlow, flowImages, frames])

  // play loop
  useEffect(() => {
    if (!playing || !ready || order.length < 2) return
    let raf = 0
    let last = performance.now()
    let acc = 0
    let p = Math.min(posRef.current, order.length - 1)
    const tick = 1000 / fps
    const loop = (now: number) => {
      acc += now - last
      last = now
      const hold = holdTicks(frames, order[p], keysOnly) * tick
      if (acc >= hold) {
        acc -= hold
        if (acc > 4 * tick) acc = 0 // tab was hidden; don't fast-forward
        p = (p + 1) % order.length
        setPos(p)
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [playing, ready, order, fps, keysOnly, frames])

  const step = useCallback(
    (d: number) => {
      setPlaying(false)
      setPos((p) => (p + d + order.length) % order.length)
    },
    [order.length],
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest('input,textarea,select,button')) return
      if (e.key === ' ') {
        e.preventDefault()
        setPlaying((p) => !p)
      } else if (e.key === 'ArrowRight') step(1)
      else if (e.key === 'ArrowLeft') step(-1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [step])

  const f = frames[current]
  return (
    <div className="space-y-4">
      <div className="relative mx-auto w-full max-w-[560px]">
        <div className="absolute inset-0 translate-x-2 translate-y-2 rotate-1 rounded-xl bg-paper-2 shadow" aria-hidden />
        <div className="absolute inset-0 translate-x-1 translate-y-1 -rotate-[0.6deg] rounded-xl bg-paper shadow" aria-hidden />
        <div className="relative overflow-hidden rounded-xl border border-line bg-white shadow-lg">
          <canvas
            ref={canvas}
            className="block h-auto w-full"
            style={{ aspectRatio: `${render.summary?.width ?? 3} / ${render.summary?.height ?? 4}` }}
            aria-label="Animation preview"
          />
          {!ready && (
            <div className="absolute inset-0 grid place-items-center bg-paper/80 text-sm text-ink-2">
              Loading frames {loaded}/{frames.length}
            </div>
          )}
          {f && (
            <span
              className={clsx(
                'absolute top-2 left-2 rounded-full px-2 py-0.5 font-mono text-[11px]',
                f.key ? 'bg-ink text-white' : 'bg-accent text-white',
              )}
            >
              {f.key ? `page ${f.source + 1}` : `t = ${f.t.toFixed(2)}`}
            </span>
          )}
        </div>
      </div>

      {/* timeline */}
      <div className="relative h-8 select-none" aria-hidden>
        <div className="absolute inset-x-0 top-1/2 h-px bg-line" />
        {frames.map((fr, i) => (
          <span
            key={i}
            className={clsx(
              'absolute top-1/2 w-[2px] -translate-x-1/2 -translate-y-1/2 rounded-full',
              fr.key ? 'h-6 bg-ink' : 'h-3 bg-rule',
              i === current && 'w-1 !bg-accent',
              keysOnly && !fr.key && 'opacity-30',
            )}
            style={{ left: `${frames.length > 1 ? (i / (frames.length - 1)) * 100 : 50}%` }}
          />
        ))}
      </div>
      <input
        type="range"
        aria-label="Scrub"
        min={0}
        max={Math.max(0, order.length - 1)}
        value={Math.min(pos, order.length - 1)}
        onChange={(e) => {
          setPlaying(false)
          setPos(Number(e.target.value))
        }}
        className="-mt-3 w-full accent-[var(--color-accent)]"
      />

      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="ghost" onClick={() => step(-1)} aria-label="Previous frame">
          <ChevronLeft className="size-4" />
        </Button>
        <Button size="sm" variant="primary" onClick={() => setPlaying(!playing)} aria-label={playing ? 'Pause' : 'Play'} className="w-20">
          {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
          {playing ? 'Pause' : 'Play'}
        </Button>
        <Button size="sm" variant="ghost" onClick={() => step(1)} aria-label="Next frame">
          <ChevronRight className="size-4" />
        </Button>
        <div className="flex items-center gap-2">
          <div className="w-32">
            <Segmented label="Playback speed" value={String(fps)} onChange={(v) => setFps(Number(v))} options={FPS.map((n) => ({ value: String(n), label: String(n) }))} />
          </div>
          <span className="text-xs text-ink-3">fps</span>
        </div>
        <span className="ml-auto font-mono text-xs text-ink-3">
          {current + 1}/{frames.length}
        </span>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <Toggle label="Pages only" hint="Hide the generated frames to compare" checked={keysOnly} onChange={restart(setKeysOnly)} />
        <Toggle label="Ping-pong" hint="Play forward, then backward" checked={pingpong} onChange={restart(setPingpong)} />
        <Toggle label="Onion skin" hint="Previous page in red, next in blue" checked={onion} onChange={setOnion} />
        <Toggle label="Show flow" hint="Colour = direction, strength = speed" checked={showFlow} onChange={setShowFlow} />
      </div>

      <div className="flex flex-wrap gap-2">
        <a className={buttonClass('secondary', 'sm')} href={api.exportUrl(render.id, 'gif', fps, pingpong)} download>
          <Download className="size-4" /> GIF
        </a>
        <a className={buttonClass('secondary', 'sm')} href={api.exportUrl(render.id, 'mp4', fps, pingpong)} download>
          <Download className="size-4" /> MP4
        </a>
      </div>
    </div>
  )
}
