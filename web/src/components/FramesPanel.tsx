import { DndContext, KeyboardSensor, PointerSensor, closestCenter, useSensor, useSensors, type DragEndEvent } from '@dnd-kit/core'
import { SortableContext, arrayMove, rectSortingStrategy, sortableKeyboardCoordinates, useSortable } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import clsx from 'clsx'
import { ImagePlus, X } from 'lucide-react'
import { useRef } from 'react'
import type { Frame } from '../api'

function Thumb({ frame, index, onDelete, disabled }: { frame: Frame; index: number; onDelete: () => void; disabled: boolean }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id: frame.id, disabled })
  return (
    <li
      ref={setNodeRef}
      style={{ transform: CSS.Transform.toString(transform), transition }}
      className={clsx('group relative', isDragging && 'z-10')}
    >
      <button
        type="button"
        {...attributes}
        {...listeners}
        aria-label={`Page ${index + 1}: ${frame.name}. Drag to reorder.`}
        className={clsx(
          'block w-full cursor-grab overflow-hidden rounded-lg border bg-white active:cursor-grabbing',
          isDragging ? 'border-accent shadow-lg' : 'border-line',
        )}
      >
        <img src={frame.thumb_url} alt="" draggable={false} className="aspect-[3/4] w-full object-cover" />
      </button>
      <span className="pointer-events-none absolute bottom-1 left-1 rounded bg-ink/75 px-1.5 font-mono text-[10px] text-white">{index + 1}</span>
      {!disabled && (
        <button
          type="button"
          onClick={onDelete}
          aria-label={`Remove page ${index + 1}`}
          className="absolute -top-2 -right-2 hidden rounded-full border border-line bg-paper p-1 text-ink-2 shadow group-hover:block hover:text-bad focus:block"
        >
          <X className="size-3" />
        </button>
      )}
    </li>
  )
}

export function FramesPanel({
  frames,
  busy,
  onReorder,
  onDelete,
  onUpload,
  allowUpload,
}: {
  frames: Frame[]
  busy: boolean
  onReorder: (ids: string[]) => void
  onDelete: (id: string) => void
  onUpload: (files: File[]) => void
  allowUpload: boolean
}) {
  const input = useRef<HTMLInputElement>(null)
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )
  const onDragEnd = (e: DragEndEvent) => {
    if (!e.over || e.active.id === e.over.id) return
    const ids = frames.map((f) => f.id)
    onReorder(arrayMove(ids, ids.indexOf(String(e.active.id)), ids.indexOf(String(e.over.id))))
  }
  return (
    <div>
      <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
        <SortableContext items={frames.map((f) => f.id)} strategy={rectSortingStrategy}>
          <ol className="grid grid-cols-4 gap-2.5 sm:grid-cols-5 lg:grid-cols-4">
            {frames.map((f, i) => (
              <Thumb key={f.id} frame={f} index={i} onDelete={() => onDelete(f.id)} disabled={busy} />
            ))}
            {allowUpload && (
              <li>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => input.current?.click()}
                  className="flex aspect-[3/4] w-full flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-ink-3/50 text-xs text-ink-2 hover:border-accent hover:text-accent disabled:opacity-50"
                >
                  <ImagePlus className="size-5" />
                  Add
                </button>
              </li>
            )}
          </ol>
        </SortableContext>
      </DndContext>
      <input
        ref={input}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(e) => {
          const files = Array.from(e.target.files ?? [])
          files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
          if (files.length) onUpload(files)
          e.target.value = ''
        }}
      />
      {frames.length > 0 && (
        <p className="mt-3 text-xs text-ink-3">
          {frames.length} page{frames.length === 1 ? '' : 's'} · drag to reorder · files are sorted by the number in their name
        </p>
      )}
    </div>
  )
}
