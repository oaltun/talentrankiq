import { Link, useNavigate, useRouterState } from '@tanstack/react-router'
import { ask } from '@tauri-apps/plugin-dialog'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { executeCommand, useCommandContext } from '@/lib/commands'
import {
  useArchiveJob,
  useDeleteJob,
  useJob,
  useJobUiState,
} from '@/services/jobs'
import { useRankJob } from '@/services/rank'
import { useTranslation } from 'react-i18next'
import {
  Archive,
  Loader2,
  MoreHorizontal,
  Settings,
  Shuffle,
  Trash2,
} from 'lucide-react'

/** Derive the current job id (if any) from the route. */
function useCurrentJobId(): string | undefined {
  const path = useRouterState({ select: s => s.location.pathname })
  if (path.startsWith('/jobs/')) return path.slice('/jobs/'.length)
  return undefined
}

/**
 * Left side of the title bar: `TalentRankIQ` wordmark +, on a job route, the
 * breadcrumb (▸ title + facts).
 */
export function TitleBarLeftActions() {
  const jobId = useCurrentJobId()
  const { data } = useJob(jobId)

  return (
    <div className="flex items-center gap-1">
      <Link
        to="/"
        className="px-1 text-sm font-bold tracking-tight text-primary"
      >
        TalentRankIQ
      </Link>

      {jobId && data && (
        <span className="flex items-center gap-1.5 text-sm text-muted-foreground">
          <span className="text-foreground/30">▸</span>
          <span className="font-medium text-foreground/90">
            {data.job.title}
          </span>
          <span className="text-xs">
            {data.cv_imported} CVs
            {data.job.last_ranked_at ? ' · ranked' : ''}
          </span>
        </span>
      )}
    </div>
  )
}

/**
 * Right side: primary action (New Job on the list, Sort on a job) + `⋯` overflow
 * (Archive / Delete on a job) + preferences.
 */
export function TitleBarRightActions() {
  const { t } = useTranslation()
  const commandContext = useCommandContext()
  const jobId = useCurrentJobId()
  const { data } = useJob(jobId)
  const { data: uiState } = useJobUiState(jobId)
  const rankJob = useRankJob(jobId ?? '__none__')
  // `ranking`, not `running`: the import auto-start embeds ad-less jobs in the
  // background (no ad = nothing to rank), which must not spin this button.
  const rankRunning = rankJob.state?.ranking === true
  const archiveJob = useArchiveJob()
  const deleteJob = useDeleteJob()
  const navigate = useNavigate()

  const handleOpenPreferences = async () => {
    const result = await executeCommand('open-preferences', commandContext)
    if (!result.success && result.error) {
      commandContext.showToast(result.error, 'error')
    }
  }

  const handleDelete = async () => {
    if (!jobId || !data) return
    const ok = await ask(
      `Delete "${data.job.title}" and all its candidates? This cannot be undone.`,
      { title: 'Delete job', kind: 'warning' }
    )
    if (ok) {
      await deleteJob.mutateAsync(jobId)
      navigate({ to: '/' })
    }
  }

  // Backend-derived enablement: the serve owns the reason (`no-cvs`,
  // `no-jobad`, …); the gui only maps it to a tooltip. While the ui-state
  // poll has not answered yet, fall back to the local structural check.
  const canSort = uiState
    ? uiState.sort_allowed
    : !!data && data.cv_imported >= 1
  const sortReason = uiState?.reason ?? (canSort ? 'clean' : 'no-cvs')
  const sortTitle = canSort
    ? t('rank.startTitle')
    : t(`rank.reason.${sortReason}`)

  return (
    <div className="flex items-center gap-1">
      {jobId && data ? (
        <>
          <Button
            size="sm"
            onClick={() => rankJob.mutate()}
            disabled={!canSort || rankJob.isPending || rankRunning}
            title={sortTitle}
          >
            {rankRunning ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Shuffle className="size-3.5" />
            )}
            {rankRunning ? t('rank.ranking') : t('rank.sort')}
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon-sm" title="More actions">
                <MoreHorizontal />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem
                onClick={() =>
                  archiveJob.mutate({ jobId, archived: !data.job.archived })
                }
              >
                <Archive className="size-4" />
                {data.job.archived ? 'Restore' : 'Archive'}
              </DropdownMenuItem>
              <DropdownMenuItem
                className="text-destructive"
                onClick={handleDelete}
              >
                <Trash2 className="size-4" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </>
      ) : null}

      <Button
        onClick={handleOpenPreferences}
        variant="ghost"
        size="icon"
        className="h-6 w-6 text-foreground/70 hover:text-foreground"
        title="Preferences"
      >
        <Settings className="h-3 w-3" />
      </Button>
    </div>
  )
}

/** The centered title slot is unused (brand lives in the left actions). */
export function TitleBarTitle(_props: { title?: string }) {
  return null
}

/** Combined toolbar content (used by the Linux title bar). */
export function TitleBarContent() {
  return (
    <>
      <TitleBarLeftActions />
      <TitleBarTitle />
      <TitleBarRightActions />
    </>
  )
}
