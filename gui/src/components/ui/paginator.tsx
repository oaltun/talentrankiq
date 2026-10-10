import { ChevronLeft, ChevronRight } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface PaginatorProps {
  page: number
  pageSize: number
  total: number
  onPageChange: (page: number) => void
  className?: string
}

/** Page controls for a result set: prev/next plus first/last + a range label. */
export function Paginator({
  page,
  pageSize,
  total,
  onPageChange,
  className,
}: PaginatorProps) {
  const pageCount = Math.max(1, Math.ceil(total / pageSize))
  const current = Math.min(page, pageCount - 1)
  const from = total === 0 ? 0 : current * pageSize + 1
  const to = Math.min(total, (current + 1) * pageSize)

  const go = (next: number) => {
    const clamped = Math.max(0, Math.min(next, pageCount - 1))
    if (clamped !== current) onPageChange(clamped)
  }

  return (
    <div
      className={cn(
        'flex items-center justify-between gap-2 border-t pt-3 text-xs text-muted-foreground',
        className
      )}
    >
      <span className="tabular-nums">
        {from}–{to} of {total}
      </span>
      <div className="flex items-center gap-1">
        <Button
          variant="outline"
          size="sm"
          className="h-7 px-2"
          onClick={() => go(0)}
          disabled={current <= 0}
          title="First page"
        >
          «
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="h-7 px-2"
          onClick={() => go(current - 1)}
          disabled={current <= 0}
          title="Previous page"
        >
          <ChevronLeft />
        </Button>
        <span className="px-1 tabular-nums">
          {current + 1}/{pageCount}
        </span>
        <Button
          variant="outline"
          size="sm"
          className="h-7 px-2"
          onClick={() => go(current + 1)}
          disabled={current >= pageCount - 1}
          title="Next page"
        >
          <ChevronRight />
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="h-7 px-2"
          onClick={() => go(pageCount - 1)}
          disabled={current >= pageCount - 1}
          title="Last page"
        >
          »
        </Button>
      </div>
    </div>
  )
}
