import { Code2, Cpu, Zap } from 'lucide-react'
import type { SystemInfo } from '../api'

export function Header({ system }: { system?: SystemInfo }) {
  const gpu = system?.cuda_device
  return (
    <header className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
      <div className="flex items-baseline gap-3">
        <h1 className="font-hand text-4xl leading-none text-ink">Flipster</h1>
        <p className="hidden text-sm text-ink-2 sm:block">Flipbook in-betweening with pyramidal Lucas–Kanade on the GPU</p>
      </div>
      <div className="flex items-center gap-2">
        {system && (
          <span
            className="inline-flex items-center gap-1.5 rounded-full border border-line bg-paper px-3 py-1 text-xs font-medium text-ink-2"
            title={`Backends: ${system.backends.join(', ')}`}
          >
            {gpu ? <Zap className="size-3.5 text-accent" /> : <Cpu className="size-3.5" />}
            {gpu ? `CUDA · ${gpu}` : system.backends.includes('cpu') ? `C++ CPU · ${system.cpu_threads ?? '?'} threads` : 'NumPy reference'}
          </span>
        )}
        <a
          href="https://github.com/lolaitan/Flipster"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm text-ink-2 hover:bg-paper hover:text-ink"
        >
          <Code2 className="size-4" /> Source
        </a>
      </div>
    </header>
  )
}
