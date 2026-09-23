import clsx from 'clsx'
import { Copy, Eraser, Layers, Pencil, Plus, Trash2, Undo2 } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Button } from './ui'

export const PAD_W = 600
export const PAD_H = 800
const SIZES = [2, 4, 8]

/**
 * A minimal flipbook page: draw, see the previous page underneath (onion skin),
 * then add the page. Exported PNGs are strokes on white; the ruled paper is only a
 * CSS background so it never ends up in the frames.
 */
export function DrawPad({ previousUrl, onAdd, busy }: { previousUrl?: string; onAdd: (png: Blob) => void; busy: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const history = useRef<ImageData[]>([])
  const last = useRef<{ x: number; y: number } | null>(null)
  const [tool, setTool] = useState<'pen' | 'eraser'>('pen')
  const [size, setSize] = useState(4)
  const [onion, setOnion] = useState(true)
  const [dirty, setDirty] = useState(false)

  const ctx = () => canvas.current!.getContext('2d', { willReadFrequently: true })!

  const paintWhite = useCallback(() => {
    const c = canvas.current!.getContext('2d', { willReadFrequently: true })!
    c.fillStyle = '#ffffff'
    c.fillRect(0, 0, PAD_W, PAD_H)
  }, [])

  useEffect(paintWhite, [paintWhite])

  const clear = () => {
    paintWhite()
    history.current = []
    setDirty(false)
  }

  const pos = (e: React.PointerEvent) => {
    const r = canvas.current!.getBoundingClientRect()
    return { x: ((e.clientX - r.left) / r.width) * PAD_W, y: ((e.clientY - r.top) / r.height) * PAD_H }
  }

  const stroke = (from: { x: number; y: number }, to: { x: number; y: number }, pressure: number) => {
    const c = ctx()
    c.lineCap = 'round'
    c.lineJoin = 'round'
    c.strokeStyle = tool === 'pen' ? '#1d1f24' : '#ffffff'
    c.lineWidth = tool === 'pen' ? size * (0.6 + 0.8 * (pressure || 0.5)) : size * 4
    c.beginPath()
    c.moveTo(from.x, from.y)
    c.lineTo(to.x, to.y)
    c.stroke()
  }

  const onDown = (e: React.PointerEvent) => {
    canvas.current!.setPointerCapture(e.pointerId)
    history.current.push(ctx().getImageData(0, 0, PAD_W, PAD_H))
    if (history.current.length > 30) history.current.shift()
    const p = pos(e)
    last.current = p
    stroke(p, { x: p.x + 0.01, y: p.y }, e.pressure)
    setDirty(true)
  }
  const onMove = (e: React.PointerEvent) => {
    if (!last.current) return
    const p = pos(e)
    stroke(last.current, p, e.pressure)
    last.current = p
  }
  const onUp = () => {
    last.current = null
  }

  const undo = () => {
    const prev = history.current.pop()
    if (prev) ctx().putImageData(prev, 0, 0)
    if (!history.current.length) setDirty(false)
  }

  const copyPrevious = () => {
    if (!previousUrl) return
    const img = new Image()
    img.onload = () => {
      history.current.push(ctx().getImageData(0, 0, PAD_W, PAD_H))
      ctx().drawImage(img, 0, 0, PAD_W, PAD_H)
      setDirty(true)
    }
    img.src = previousUrl
  }

  const add = () => {
    canvas.current!.toBlob((b) => {
      if (b) {
        onAdd(b)
        clear()
      }
    }, 'image/png')
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <Button size="sm" variant={tool === 'pen' ? 'primary' : 'secondary'} onClick={() => setTool('pen')} aria-label="Pen">
          <Pencil className="size-4" />
        </Button>
        <Button size="sm" variant={tool === 'eraser' ? 'primary' : 'secondary'} onClick={() => setTool('eraser')} aria-label="Eraser">
          <Eraser className="size-4" />
        </Button>
        <div className="mx-1 flex items-center gap-1" role="radiogroup" aria-label="Brush size">
          {SIZES.map((s) => (
            <button
              key={s}
              type="button"
              role="radio"
              aria-checked={size === s}
              aria-label={`Brush ${s}px`}
              onClick={() => setSize(s)}
              className={clsx('grid size-8 place-items-center rounded-md', size === s ? 'bg-paper-2 ring-1 ring-line' : 'hover:bg-paper-2')}
            >
              <span className="rounded-full bg-ink" style={{ width: s + 2, height: s + 2 }} />
            </button>
          ))}
        </div>
        <Button size="sm" variant="ghost" onClick={undo} aria-label="Undo">
          <Undo2 className="size-4" />
        </Button>
        <Button size="sm" variant="ghost" onClick={clear} aria-label="Clear page">
          <Trash2 className="size-4" />
        </Button>
        <span className="flex-1" />
        <Button size="sm" variant={onion ? 'secondary' : 'ghost'} onClick={() => setOnion(!onion)} disabled={!previousUrl} title="Show the previous page underneath">
          <Layers className="size-4" /> Onion
        </Button>
        <Button size="sm" variant="ghost" onClick={copyPrevious} disabled={!previousUrl} title="Start from a copy of the previous page">
          <Copy className="size-4" /> Trace
        </Button>
      </div>

      <div className="ruled relative mx-auto aspect-[3/4] w-full max-w-[420px] overflow-hidden rounded-lg border border-line shadow-inner">
        {onion && previousUrl && (
          <img src={previousUrl} alt="" className="pointer-events-none absolute inset-0 size-full opacity-25 mix-blend-multiply" />
        )}
        <canvas
          ref={canvas}
          width={PAD_W}
          height={PAD_H}
          aria-label="Drawing page"
          className="absolute inset-0 size-full touch-none mix-blend-multiply"
          style={{ cursor: 'crosshair' }}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
        />
      </div>

      <Button variant="primary" className="w-full" onClick={add} disabled={!dirty || busy}>
        <Plus className="size-4" /> Add page
      </Button>
    </div>
  )
}
