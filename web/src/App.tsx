import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { FilePlus2, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api, DEFAULT_RENDER, type Project, type RenderRequest, type Source } from './api'
import { DrawPad } from './components/DrawPad'
import { EmptyState } from './components/EmptyState'
import { FramesPanel } from './components/FramesPanel'
import { Header } from './components/Header'
import { Player } from './components/Player'
import { SettingsPanel } from './components/SettingsPanel'
import { StatsPanel } from './components/StatsPanel'
import { Button, Card } from './components/ui'
import { useJobEvents } from './hooks/useJobEvents'

const STORAGE_KEY = 'flipster.project'

function remembered(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

function remember(id: string | null) {
  try {
    if (id) localStorage.setItem(STORAGE_KEY, id)
    else localStorage.removeItem(STORAGE_KEY)
  } catch {
    /* storage unavailable (private mode) - not important */
  }
}

export default function App() {
  const qc = useQueryClient()
  const [projectId, setProjectId] = useState<string | null>(remembered)
  const [tab, setTab] = useState<'pages' | 'draw'>('pages')
  const [settings, setSettings] = useState<RenderRequest>(DEFAULT_RENDER)
  const [jobId, setJobId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dismissedJob, setDismissedJob] = useState<string | null>(null)
  const uploadInput = useRef<HTMLInputElement>(null)

  const system = useQuery({ queryKey: ['system'], queryFn: api.system })
  const project = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => api.project(projectId!),
    enabled: !!projectId,
    retry: false,
  })
  const job = useJobEvents(jobId)
  const renderId = job?.status === 'done' ? job.render_id : null
  const render = useQuery({ queryKey: ['render', renderId], queryFn: () => api.render(renderId!), enabled: !!renderId })
  const jobError = job?.status === 'error' && job.id !== dismissedJob ? (job.error ?? 'Render failed') : null
  const shownError = error ?? jobError

  useEffect(() => remember(projectId), [projectId])

  const setProject = (p: Project) => qc.setQueryData(['project', p.id], p)
  const fail = (e: unknown) => setError(e instanceof Error ? e.message : String(e))

  const startRender = useMutation({
    mutationFn: ({ id, body }: { id: string; body: RenderRequest }) => api.startRender(id, body),
    onSuccess: (r) => setJobId(r.job_id),
    onError: fail,
  })

  const newProject = useMutation({
    mutationFn: async ({ source, files, samples }: { source: Source; files?: File[]; samples?: boolean }) => {
      let p = await api.createProject(source)
      if (samples) p = await api.loadSamples(p.id)
      if (files?.length) p = await api.uploadFrames(p.id, files)
      return { p, samples }
    },
    onSuccess: ({ p, samples }) => {
      setProject(p)
      setProjectId(p.id)
      setJobId(null)
      setTab(p.source === 'drawing' ? 'draw' : 'pages')
      setSettings((s) => ({ ...s, style: 'clean' }))
      if (samples) startRender.mutate({ id: p.id, body: settings })
    },
    onError: fail,
  })

  const upload = useMutation({
    mutationFn: (files: Blob[]) => api.uploadFrames(projectId!, files),
    onSuccess: setProject,
    onError: fail,
  })
  const reorder = useMutation({
    mutationFn: (ids: string[]) => api.reorder(projectId!, ids),
    onMutate: (ids) => {
      const p = qc.getQueryData<Project>(['project', projectId])
      if (p) setProject({ ...p, frames: ids.map((id) => p.frames.find((f) => f.id === id)!) })
    },
    onSuccess: setProject,
    onError: fail,
  })
  const remove = useMutation({ mutationFn: (fid: string) => api.deleteFrame(projectId!, fid), onSuccess: setProject, onError: fail })

  const p = project.isError ? undefined : project.data // an expired project falls back to the start screen
  const frames = p?.frames ?? []
  const busy = newProject.isPending || upload.isPending
  const rendering = job?.status === 'queued' || job?.status === 'running'
  const hasProject = !!p

  const reset = () => {
    setProjectId(null)
    setJobId(null)
  }

  return (
    <div className="flex min-h-full flex-col">
      <Header system={system.data} />

      {shownError && (
        <div role="alert" className="mx-auto mb-3 flex w-full max-w-7xl items-start gap-3 px-4 sm:px-6">
          <div className="flex w-full items-start justify-between gap-3 rounded-xl border border-bad/30 bg-bad/10 px-4 py-2.5 text-sm text-bad">
            {shownError}
            <button
              type="button"
              aria-label="Dismiss"
              onClick={() => {
                setError(null)
                if (job) setDismissedJob(job.id)
              }}
            >
              <X className="size-4" />
            </button>
          </div>
        </div>
      )}

      <main className="mx-auto grid w-full max-w-7xl flex-1 gap-5 px-4 pb-10 sm:px-6 lg:grid-cols-[400px_1fr]">
        <aside className="order-2 space-y-5 lg:order-none">
          <Card
            title={
              hasProject ? (
                p.source === 'drawing' ? (
                  <span className="flex gap-1">
                    {(['draw', 'pages'] as const).map((t) => (
                      <button
                        key={t}
                        type="button"
                        onClick={() => setTab(t)}
                        className={clsx('rounded-md px-2 py-0.5', tab === t ? 'bg-paper-2 text-ink' : 'text-ink-3 hover:text-ink')}
                      >
                        {t === 'draw' ? 'Draw' : `Pages (${frames.length})`}
                      </button>
                    ))}
                  </span>
                ) : (
                  `Pages (${frames.length})`
                )
              ) : (
                'Pages'
              )
            }
            action={
              hasProject && (
                <Button size="sm" variant="ghost" onClick={reset} title="Start over">
                  <FilePlus2 className="size-4" /> New
                </Button>
              )
            }
          >
            {!hasProject ? (
              <p className="text-sm text-ink-2">Start with the sample, your own scans, or draw pages right here.</p>
            ) : p.source === 'drawing' && tab === 'draw' ? (
              <DrawPad
                busy={upload.isPending}
                previousUrl={frames.at(-1)?.image_url}
                onAdd={(png) => upload.mutate([new File([png], `page-${frames.length + 1}.png`, { type: 'image/png' })])}
              />
            ) : frames.length === 0 ? (
              <p className="text-sm text-ink-2">No pages yet.</p>
            ) : (
              <FramesPanel
                frames={frames}
                busy={busy || rendering}
                onReorder={(ids) => reorder.mutate(ids)}
                onDelete={(id) => remove.mutate(id)}
                onUpload={(files) => upload.mutate(files)}
                allowUpload={p.source !== 'drawing'}
              />
            )}
          </Card>

          <Card title="Render">
            <SettingsPanel
              value={settings}
              onChange={setSettings}
              system={system.data}
              source={p?.source ?? 'scan'}
              canRender={frames.length >= 2}
              job={job}
              onRender={() => p && startRender.mutate({ id: p.id, body: settings })}
            />
          </Card>
        </aside>

        <section className="order-1 min-w-0 space-y-5 lg:order-none">
          {hasProject && frames.length === 0 ? (
            <div className="ruled grid min-h-[420px] place-items-center rounded-2xl border border-line p-10 text-center shadow-lg">
              <div>
                <p className="font-hand text-3xl text-ink-2">{p.source === 'drawing' ? 'draw your first page' : 'add some pages'}</p>
                <p className="mt-2 text-sm text-ink-3">
                  {p.source === 'drawing'
                    ? 'Sketch on the page to the left, then “Add page”. Turn on onion skin to trace the next one.'
                    : 'Upload photos or scans of your flipbook pages.'}
                </p>
              </div>
            </div>
          ) : !hasProject ? (
            <EmptyState
              busy={busy}
              samplesAvailable={system.data?.samples ?? false}
              onSample={() => newProject.mutate({ source: 'scan', samples: true })}
              onUpload={() => uploadInput.current?.click()}
              onDraw={() => newProject.mutate({ source: 'drawing' })}
            />
          ) : render.data && render.data.project_id === p.id ? (
            <>
              <Card title="Animation">
                <Player key={render.data.id} render={render.data} />
              </Card>
              {render.data.summary && (
                <Card title="Performance">
                  <StatsPanel summary={render.data.summary} inbetweens={render.data.options.method === 'none' ? 0 : render.data.options.inbetweens} />
                </Card>
              )}
            </>
          ) : (
            <div className="ruled grid min-h-[420px] place-items-center rounded-2xl border border-line p-10 text-center shadow-lg">
              <div>
                <img src={frames[0].thumb_url} alt="" className="mx-auto mb-5 w-40 rotate-[-2deg] rounded-md border border-line shadow-md" />
                <p className="font-hand text-3xl text-ink-2">{rendering ? 'drawing the in-betweens…' : `${frames.length} page${frames.length === 1 ? '' : 's'} ready`}</p>
                <p className="mt-2 text-sm text-ink-3">
                  {frames.length < 2 ? 'Add at least two pages to animate.' : rendering ? job?.message : 'Hit “Render animation” to generate the in-betweens.'}
                </p>
              </div>
            </div>
          )}
        </section>
      </main>

      <footer className="pb-6 text-center text-xs text-ink-3">
        Flipster v2 · React + FastAPI + C++/CUDA ·{' '}
        <a className="underline hover:text-ink" href="https://github.com/lolaitan/Flipster">
          source
        </a>
      </footer>

      <input
        ref={uploadInput}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
          if (files.length) newProject.mutate({ source: 'scan', files })
          e.target.value = ''
        }}
      />
    </div>
  )
}
