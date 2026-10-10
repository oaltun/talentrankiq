import { toast } from 'sonner'
import { openPath } from '@tauri-apps/plugin-opener'

import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { errorMessage } from '@/lib/errors'
import { openFileDialog } from '@/lib/file-dialogs'
import { useJobAdFile, useSetJobAdPath } from '@/services/jobs'
import { FileText, FolderOpen } from 'lucide-react'

import { PaneCard } from './PaneCard'
import { useJobAdReloaded } from './useJobAdReloaded'

/**
 * The job-ad file's controls: where it is, point it at a different one, open it
 * in your editor, and a read-only 3-line mirror of its current contents. Leads
 * the Setup pane.
 *
 * Owns the single `job-ad-reloaded` subscription, which re-reads the file and
 * refreshes the mirror below from the same query: one listener, one
 * invalidation, cannot drift.
 */
export function AdFileCard({ jobId }: { jobId: string }) {
  const { data: adFile } = useJobAdFile(jobId)
  const setAdPath = useSetJobAdPath(jobId)
  useJobAdReloaded(jobId)

  const path = adFile?.path ?? ''
  const text = adFile?.text ?? ''

  const handleChange = async () => {
    const chosen = await openFileDialog({
      title: 'Choose the job-ad text file',
      startDir: path || undefined,
      filters: [
        { name: 'Text', extensions: ['txt', 'md', 'markdown', 'text'] },
      ],
    })
    if (typeof chosen !== 'string') return
    setAdPath.mutate(chosen, {
      onSuccess: () => toast.success('Job-ad file changed'),
      onError: e =>
        toast.error(`Could not change the job-ad file: ${errorMessage(e)}`),
    })
  }

  /** Reveal the ad in the system editor — the default handler for `.txt`. */
  const handleOpen = async () => {
    if (!path) return
    try {
      await openPath(path)
    } catch (e) {
      toast.error(`Could not open the job ad: ${errorMessage(e)}`)
    }
  }

  return (
    <PaneCard title="Ad file">
      <div className="flex flex-col gap-1">
        <div className="min-w-0 cursor-text rounded-md border px-3 py-1 font-mono text-xs break-all text-muted-foreground select-text">
          {path || <span className="italic">No file selected</span>}
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleChange}
            disabled={setAdPath.isPending}
            title="Point the job at a different text file"
          >
            <FolderOpen className="size-3.5" />
            Change
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleOpen}
            disabled={!path}
            title="Open the job ad in your text editor"
          >
            <FileText className="size-3.5" />
            Open
          </Button>
        </div>

        <Textarea
          value={text}
          readOnly
          rows={3}
          className="h-[76px] min-h-0 resize-none text-xs font-mono"
          aria-label="Job-ad file contents (read-only)"
        />
        {text.trim() || path ? null : (
          <span className="italic text-xs text-muted-foreground">
            No file selected.
          </span>
        )}
        {adFile?.created && (
          <p className="text-xs text-muted-foreground">
            Created an empty file — write your ad there and save; the ad is
            re-read and the job re-ranked automatically.
          </p>
        )}
      </div>
    </PaneCard>
  )
}
