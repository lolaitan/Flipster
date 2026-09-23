import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { DEFAULT_RENDER, type Job, type SystemInfo } from '../api'
import { SettingsPanel } from './SettingsPanel'

const system: SystemInfo = {
  backends: ['cuda', 'cpu', 'numpy'],
  cuda_device: 'RTX',
  cuda_compiled: true,
  cpu_threads: 8,
  samples: true,
  limits: { max_frames: 60, max_upload_mb: 25, max_working_height: 1600 },
}

function setup(overrides: Partial<Parameters<typeof SettingsPanel>[0]> = {}) {
  const props = {
    value: DEFAULT_RENDER,
    onChange: vi.fn(),
    onRender: vi.fn(),
    system,
    source: 'scan' as const,
    canRender: true,
    job: null,
    ...overrides,
  }
  render(<SettingsPanel {...props} />)
  return props
}

describe('SettingsPanel', () => {
  it('switches interpolation method', async () => {
    const props = setup()
    await userEvent.click(screen.getByRole('radio', { name: 'Cross-fade' }))
    expect(props.onChange).toHaveBeenCalledWith({ ...DEFAULT_RENDER, method: 'linear' })
  })

  it('lists the server backends', () => {
    setup()
    expect(screen.getByRole('radio', { name: 'CUDA' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'C++ CPU' })).toBeInTheDocument()
  })

  it('hides scan-only options for drawings', () => {
    setup({ source: 'drawing' })
    expect(screen.queryByText('Stabilise pages')).not.toBeInTheDocument()
  })

  it('disables rendering while a job runs and shows progress', () => {
    const job: Job = { id: 'j', render_id: 'r', project_id: 'p', status: 'running', stage: 'interpolate', progress: 0.5, message: 'Pair 3/19', error: null }
    setup({ job })
    expect(screen.getByRole('button', { name: /rendering/i })).toBeDisabled()
    expect(screen.getByText('Pair 3/19')).toBeInTheDocument()
  })

  it('requires two pages', async () => {
    const props = setup({ canRender: false })
    await userEvent.click(screen.getByRole('button', { name: /render animation/i }))
    expect(props.onRender).not.toHaveBeenCalled()
  })
})
