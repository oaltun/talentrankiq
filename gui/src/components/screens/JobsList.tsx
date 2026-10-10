import { useNavigate } from '@tanstack/react-router'
import { ask } from '@tauri-apps/plugin-dialog'

import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  useArchiveJob,
  useCreateJob,
  useDeleteJob,
  useJobs,
} from '@/services/jobs'
import type { JobSummary } from '@/lib/tauri-bindings'
import { Loader2, MoreHorizontal, Plus } from 'lucide-react'

function relativeDay(iso: string): string {
  // SQLite datetime('now') is UTC "YYYY-MM-DD HH:MM:SS".
  const then = new Date(`${iso.replace(' ', 'T')}Z`).getTime()
  const days = Math.floor((Date.now() - then) / 86_400_000)
  if (days <= 0) return 'today'
  if (days === 1) return '1d ago'
  return `${days}d ago`
}

function JobCard({ job }: { job: JobSummary }) {
  const navigate = useNavigate()
  const archiveJob = useArchiveJob()
  const deleteJob = useDeleteJob()

  const open = () => navigate({ to: '/jobs/$jobId', params: { jobId: job.id } })

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation()
    const ok = await ask(
      `Delete "${job.title}" and all its candidates? This cannot be undone.`,
      { title: 'Delete job', kind: 'warning' }
    )
    if (ok) await deleteJob.mutateAsync(job.id)
  }

  return (
    <div
      onClick={open}
      className="group flex cursor-pointer items-center gap-3 rounded-lg border bg-card p-4 transition-colors hover:bg-accent/40"
    >
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate font-medium">{job.title}</span>
          {job.archived && <Badge variant="secondary">Archived</Badge>}
        </div>
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
          <Badge variant="outline">{job.cv_imported} CVs</Badge>
          {job.cv_failed > 0 && (
            <Badge variant="outline">{job.cv_failed} failed</Badge>
          )}
          {job.last_ranked_at && (
            <span>ranked {relativeDay(job.last_ranked_at)}</span>
          )}
        </div>
      </div>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={e => e.stopPropagation()}
            className="opacity-0 group-hover:opacity-100"
          >
            <MoreHorizontal />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem
            onClick={e => {
              e.stopPropagation()
              archiveJob.mutate({ jobId: job.id, archived: !job.archived })
            }}
          >
            {job.archived ? 'Restore' : 'Archive'}
          </DropdownMenuItem>
          <DropdownMenuItem className="text-destructive" onClick={handleDelete}>
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}

export function JobsList() {
  const { data: jobs, isLoading } = useJobs()
  const createJob = useCreateJob()
  const navigate = useNavigate()

  const handleNew = async () => {
    const job = await createJob.mutateAsync()
    navigate({ to: '/jobs/$jobId', params: { jobId: job.id } })
  }

  if (isLoading) {
    return (
      <div className="flex flex-1 items-center justify-center text-muted-foreground">
        <Loader2 className="size-5 animate-spin" />
      </div>
    )
  }

  const active = jobs?.filter(j => !j.archived) ?? []

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col overflow-auto p-6">
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-lg font-semibold">Jobs</h1>
        <Button onClick={handleNew} disabled={createJob.isPending}>
          <Plus className="size-4" />
          New Job
        </Button>
      </div>

      {active.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed py-20 text-center">
          <p className="text-muted-foreground">No jobs yet.</p>
          <Button onClick={handleNew} disabled={createJob.isPending}>
            <Plus className="size-4" />
            New Job
          </Button>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {active.map(job => (
            <JobCard key={job.id} job={job} />
          ))}
        </div>
      )}
    </div>
  )
}
