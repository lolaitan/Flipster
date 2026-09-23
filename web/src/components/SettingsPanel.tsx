import { ChevronDown, Sparkles } from 'lucide-react'
import { useState } from 'react'
import type { Job, RenderRequest, Source, SystemInfo } from '../api'
import { Button, Field, Range, Segmented, Toggle } from './ui'

const BACKEND_LABEL: Record<string, string> = { auto: 'Auto', cuda: 'CUDA', cpu: 'C++ CPU', numpy: 'NumPy' }

export function SettingsPanel({
  value,
  onChange,
  onRender,
  system,
  source,
  canRender,
  job,
}: {
  value: RenderRequest
  onChange: (v: RenderRequest) => void
  onRender: () => void
  system?: SystemInfo
  source: Source
  canRender: boolean
  job: Job | null
}) {
  const [advanced, setAdvanced] = useState(false)
  const set = <K extends keyof RenderRequest>(k: K, v: RenderRequest[K]) => onChange({ ...value, [k]: v })
  const setFlow = <K extends keyof RenderRequest['flow']>(k: K, v: RenderRequest['flow'][K]) =>
    onChange({ ...value, flow: { ...value.flow, [k]: v } })
  const running = job?.status === 'queued' || job?.status === 'running'
  const backends = ['auto', ...(system?.backends ?? [])]

  return (
    <div className="space-y-4">
      <Field label="In-betweening">
        <Segmented
          label="Interpolation method"
          value={value.method}
          onChange={(m) => set('method', m)}
          options={[
            { value: 'none', label: 'Pages only' },
            { value: 'linear', label: 'Cross-fade', hint: 'Plain blending: what naive interpolation looks like' },
            { value: 'flow', label: 'Optical flow', hint: 'Pyramidal Lucas–Kanade + occlusion-aware splatting' },
          ]}
        />
      </Field>

      <Field label="Frames between pages" value={value.method === 'none' ? '—' : value.inbetweens}>
        <Range label="Frames between pages" min={1} max={7} value={value.inbetweens} onChange={(n) => set('inbetweens', n)} />
      </Field>

      {source === 'scan' && (
        <>
          <Field label="Look">
            <Segmented
              label="Output style"
              value={value.style}
              onChange={(s) => set('style', s)}
              options={[
                { value: 'clean', label: 'Clean ink', hint: 'Ink extracted onto blank paper' },
                { value: 'photo', label: 'Original scan', hint: 'Keep the notebook paper' },
              ]}
            />
          </Field>
          <Toggle
            label="Stabilise pages"
            hint="Align scans on the printed ruled lines before tracking"
            checked={value.stabilize}
            onChange={(v) => set('stabilize', v)}
          />
        </>
      )}

      <Field label="Compute backend">
        <Segmented
          label="Compute backend"
          value={value.backend}
          onChange={(b) => set('backend', b)}
          options={backends.map((b) => ({ value: b, label: BACKEND_LABEL[b] ?? b }))}
        />
      </Field>

      <button
        type="button"
        onClick={() => setAdvanced(!advanced)}
        className="flex w-full items-center justify-between text-xs font-semibold tracking-wide text-ink-3 uppercase hover:text-ink"
        aria-expanded={advanced}
      >
        Lucas–Kanade parameters
        <ChevronDown className={`size-4 transition-transform ${advanced ? 'rotate-180' : ''}`} />
      </button>
      {advanced && (
        <div className="space-y-3 rounded-xl bg-paper-2 p-3">
          <Field label="Pyramid levels" value={value.flow.levels === 0 ? 'auto' : value.flow.levels}>
            <Range label="Pyramid levels" min={0} max={7} value={value.flow.levels} onChange={(n) => setFlow('levels', n)} />
          </Field>
          <Field label="Iterations per level" value={value.flow.iterations}>
            <Range label="Iterations per level" min={1} max={12} value={value.flow.iterations} onChange={(n) => setFlow('iterations', n)} />
          </Field>
          <Field label="Window" value={`${2 * value.flow.window_radius + 1}×${2 * value.flow.window_radius + 1}`}>
            <Range label="Window radius" min={2} max={12} value={value.flow.window_radius} onChange={(n) => setFlow('window_radius', n)} />
          </Field>
          <Field label="Working height" value={`${value.working_height}px`}>
            <Range
              label="Working height"
              min={480}
              max={system?.limits.max_working_height ?? 1600}
              step={80}
              value={value.working_height}
              onChange={(n) => set('working_height', n)}
            />
          </Field>
          <Toggle label="Median filter" hint="3×3 median on the flow after each level" checked={value.flow.median} onChange={(v) => setFlow('median', v)} />
        </div>
      )}

      <Button variant="primary" className="w-full" onClick={onRender} disabled={!canRender || running}>
        <Sparkles className="size-4" />
        {running ? 'Rendering…' : 'Render animation'}
      </Button>

      {job && (
        <div aria-live="polite" className="space-y-1.5">
          <div className="h-1.5 overflow-hidden rounded-full bg-paper-2">
            <div
              className={`h-full rounded-full transition-[width] duration-200 ${job.status === 'error' ? 'bg-bad' : 'bg-accent'}`}
              style={{ width: `${Math.round((job.status === 'done' ? 1 : job.progress) * 100)}%` }}
            />
          </div>
          <p className={`text-xs ${job.status === 'error' ? 'text-bad' : 'text-ink-2'}`}>{job.error ?? job.message}</p>
        </div>
      )}
    </div>
  )
}
