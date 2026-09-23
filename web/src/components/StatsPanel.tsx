import type { RenderSummary } from '../api'
import { formatMs } from '../lib/playback'
import { Stat } from './ui'

const BACKEND: Record<string, string> = { cuda: 'CUDA', 'cuda-naive': 'CUDA (naive)', cpu: 'C++ / OpenMP', numpy: 'NumPy', none: '—' }

export function StatsPanel({ summary, inbetweens }: { summary: RenderSummary; inbetweens: number }) {
  const pairs = summary.pairs
  const flowPerPair = pairs.length ? summary.flow_ms / pairs.length : 0
  const synthPerFrame = pairs.length && inbetweens ? summary.synth_ms / (pairs.length * inbetweens) : 0
  const motion = pairs.length ? Math.max(...pairs.map((p) => p.mean_motion_px)) : 0
  const maxFlow = Math.max(1, ...pairs.map((p) => p.flow_ms))
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Backend" value={BACKEND[summary.backend] ?? summary.backend} sub={`${summary.width}×${summary.height}`} />
        <Stat label="Flow / pair" value={flowPerPair ? formatMs(flowPerPair) : '—'} sub="forward + backward" />
        <Stat label="Synthesis" value={synthPerFrame ? formatMs(synthPerFrame) : '—'} sub="per in-between" />
        <Stat label="Total" value={formatMs(summary.total_ms)} sub={`${summary.frames} frames`} />
      </div>
      {pairs.some((p) => p.flow_ms > 0) && (
        <div>
          <div className="mb-1.5 flex items-baseline justify-between text-xs text-ink-3">
            <span>Flow time per page pair</span>
            <span>largest mean ink motion {motion.toFixed(0)} px</span>
          </div>
          <div className="flex h-12 items-end gap-[3px]" role="img" aria-label="Flow time per page pair">
            {pairs.map((p) => (
              <div
                key={p.pair}
                className="flex-1 rounded-t-[3px] bg-rule hover:bg-accent"
                style={{ height: `${Math.max(6, (p.flow_ms / maxFlow) * 100)}%` }}
                title={`Pages ${p.pair + 1}→${p.pair + 2}: flow ${formatMs(p.flow_ms)}, ${p.mean_motion_px.toFixed(1)} px mean motion`}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
