import { useEffect, useState } from 'react'
import { api, type Job } from '../api'

/**
 * Follows a render job over server-sent events, falling back to polling if the
 * stream drops (some proxies buffer SSE).
 */
export function useJobEvents(jobId: string | null): Job | null {
  const [job, setJob] = useState<Job | null>(null)

  useEffect(() => {
    if (!jobId) return
    let closed = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const finished = (j: Job) => j.status === 'done' || j.status === 'error'

    const poll = async () => {
      if (closed) return
      try {
        const j = await api.job(jobId)
        setJob(j)
        if (finished(j)) return
      } catch {
        /* transient; retry */
      }
      timer = setTimeout(poll, 400)
    }

    const es = new EventSource(api.jobEventsUrl(jobId))
    es.onmessage = (ev) => {
      const j = JSON.parse(ev.data) as Job
      setJob(j)
      if (finished(j)) es.close()
    }
    es.onerror = () => {
      es.close()
      if (!closed) poll()
    }
    return () => {
      closed = true
      es.close()
      if (timer) clearTimeout(timer)
    }
  }, [jobId])

  return jobId && job?.id === jobId ? job : null
}
