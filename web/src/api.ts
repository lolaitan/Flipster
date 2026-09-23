// Typed client for the FastAPI backend (server/app/main.py).

export type Source = 'scan' | 'drawing' | 'photo'
export type Method = 'none' | 'linear' | 'flow'
export type Style = 'clean' | 'photo'

export interface Frame {
  id: string
  name: string
  width: number
  height: number
  image_url: string
  thumb_url: string
}

export interface Project {
  id: string
  source: Source
  created_at: number
  frames: Frame[]
}

export interface SystemInfo {
  backends: string[]
  cuda_device: string | null
  cuda_compiled: boolean
  cpu_threads: number | null
  samples: boolean
  limits: { max_frames: number; max_upload_mb: number; max_working_height: number }
}

export interface FlowSettings {
  levels: number
  iterations: number
  window_radius: number
  damping: number
  zero_pull: number
  median: boolean
}

export interface RenderRequest {
  method: Method
  inbetweens: number
  style: Style
  backend: string
  stabilize: boolean
  working_height: number
  flow: FlowSettings
}

export type JobStatus = 'queued' | 'running' | 'done' | 'error'

export interface Job {
  id: string
  render_id: string
  project_id: string
  status: JobStatus
  stage: string
  progress: number
  message: string
  error: string | null
}

export interface PairStats {
  pair: number
  flow_ms: number
  synth_ms: number
  mean_motion_px: number
}

export interface RenderSummary {
  backend: string
  width: number
  height: number
  frames: number
  preprocess_ms: number
  flow_ms: number
  synth_ms: number
  total_ms: number
  pairs: PairStats[]
}

export interface RenderFrame {
  index: number
  key: boolean
  source: number
  t: number
  url: string
}

export interface Render {
  id: string
  project_id: string
  status: string
  options: RenderRequest
  summary: RenderSummary | null
  frames: RenderFrame[]
  flow_urls: string[]
}

export const DEFAULT_FLOW: FlowSettings = {
  levels: 0,
  iterations: 5,
  window_radius: 7,
  damping: 0.05,
  zero_pull: 0,
  median: true,
}

export const DEFAULT_RENDER: RenderRequest = {
  method: 'flow',
  inbetweens: 3,
  style: 'clean',
  backend: 'auto',
  stabilize: true,
  working_height: 1200,
  flow: DEFAULT_FLOW,
}

export class ApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail)
  }
  return (await res.json()) as T
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const api = {
  system: () => request<SystemInfo>('/api/system'),
  createProject: (source: Source) => request<Project>('/api/projects', json('POST', { source })),
  project: (id: string) => request<Project>(`/api/projects/${id}`),
  uploadFrames: (id: string, files: Blob[], names?: string[]) => {
    const form = new FormData()
    files.forEach((f, i) => form.append('files', f, names?.[i] ?? (f instanceof File ? f.name : `frame-${i}.png`)))
    return request<Project>(`/api/projects/${id}/frames`, { method: 'POST', body: form })
  },
  loadSamples: (id: string) => request<Project>(`/api/projects/${id}/samples`, { method: 'POST' }),
  reorder: (id: string, order: string[]) => request<Project>(`/api/projects/${id}/frames/order`, json('PUT', { order })),
  deleteFrame: (id: string, frameId: string) =>
    request<Project>(`/api/projects/${id}/frames/${frameId}`, { method: 'DELETE' }),
  startRender: (id: string, body: RenderRequest) =>
    request<{ job_id: string; render_id: string }>(`/api/projects/${id}/renders`, json('POST', body)),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  jobEventsUrl: (id: string) => `/api/jobs/${id}/events`,
  render: (id: string) => request<Render>(`/api/renders/${id}`),
  exportUrl: (id: string, format: 'gif' | 'mp4', fps: number, pingpong: boolean) =>
    `/api/renders/${id}/export?format=${format}&fps=${fps}&pingpong=${pingpong}`,
}
