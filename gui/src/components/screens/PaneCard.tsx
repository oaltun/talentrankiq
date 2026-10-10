import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

/**
 * One visually separated sub-panel of a workspace side pane. Shared by the
 * Setup pane (Ad file, Ad contents, CV directory, Importing, Additional
 * criteria, Scoring & rubric), so every group of controls in the workspace is
 * carded the same way.
 *
 * `action` is an optional control at the right of the title row — for a card
 * whose header owns one affordance (e.g. the Scoring card's Edit button).
 *
 * `className` is for the rare card that has to grow into the pane's leftover
 * height instead of hugging its content.
 */
export function PaneCard({
  title,
  action,
  className,
  children,
}: {
  title: string
  action?: ReactNode
  className?: string
  children: ReactNode
}) {
  return (
    <div
      className={cn(
        'flex flex-col gap-1.5 rounded-lg border bg-card p-3',
        className
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {title}
        </span>
        {action}
      </div>
      {children}
    </div>
  )
}
