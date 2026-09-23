import { BookOpen, Pencil, Upload } from 'lucide-react'
import { Button } from './ui'

export function EmptyState({
  onSample,
  onUpload,
  onDraw,
  samplesAvailable,
  busy,
}: {
  onSample: () => void
  onUpload: () => void
  onDraw: () => void
  samplesAvailable: boolean
  busy: boolean
}) {
  return (
    <div className="ruled relative overflow-hidden rounded-2xl border border-line px-8 py-12 pl-16 shadow-lg sm:px-14 sm:pl-20">
      <p className="font-hand text-3xl text-ink-2">a digital flipbook</p>
      <h2 className="mt-2 max-w-md text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
        Draw a few pages. Get the frames in between.
      </h2>
      <p className="mt-4 max-w-lg text-[15px] leading-relaxed text-ink-2">
        Flipster tracks every pencil stroke between two pages with dense pyramidal Lucas–Kanade optical flow (C++ and
        CUDA), then splats both pages forward in time to draw the in-betweens. Scans of real notebook pages are
        cleaned up and lined up on the paper first.
      </p>
      <div className="mt-8 flex flex-wrap gap-3">
        {samplesAvailable && (
          <Button variant="primary" onClick={onSample} disabled={busy}>
            <BookOpen className="size-4" /> Try the sample flipbook
          </Button>
        )}
        <Button onClick={onUpload} disabled={busy}>
          <Upload className="size-4" /> Upload scans
        </Button>
        <Button onClick={onDraw} disabled={busy}>
          <Pencil className="size-4" /> Draw pages
        </Button>
      </div>
      <p className="mt-6 text-xs text-ink-3">The sample is the 20-page stick-figure flipbook from the original 15-112 version of this project.</p>
    </div>
  )
}
