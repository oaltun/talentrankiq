import { useNavigate, useRouterState } from '@tanstack/react-router'
import { Boxes, Home, LayoutPanelLeft, Menu } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { popupAppMenu } from '@/lib/menu'
import { usePlatform } from '@/hooks/use-platform'
import { useJobs } from '@/services/jobs'
import { useUIStore } from '@/store/ui-store'

type ItemKey = 'home' | 'workspace' | 'models'

const ITEMS: {
  key: ItemKey
  label: string
  icon: typeof Home
}[] = [
  { key: 'home', label: 'Home', icon: Home },
  { key: 'workspace', label: 'Workspace', icon: LayoutPanelLeft },
  { key: 'models', label: 'Models', icon: Boxes },
]

/**
 * VS Code-style vertical activity bar on the far left. Switches between the
 * Home (job list), Workspace (current job's setup + candidates), and Models views.
 *
 * The `S` wordmark and the hamburger that pops up the app menu are
 * Linux/Windows-only: on macOS the app name sits in the native title bar and
 * the same menu items live in the system menubar (`src/lib/menu.ts`).
 */
export function ActivityBar() {
  const navigate = useNavigate()
  const pathname = useRouterState({ select: s => s.location.pathname })
  const currentJobId = useUIStore(s => s.currentJobId)
  const { data: jobs } = useJobs()
  const isMacOS = usePlatform() === 'macos'

  const handleMenuClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    // Open the menu below the hamburger, anchored at its right edge and nudged
    // right+south into the content pane so it never overlaps the activity bar
    // column (hamburger + Home/Workspace/Models icons).
    popupAppMenu({ x: rect.right + 38, y: rect.bottom + 36 })
  }

  const activeKey = (): ItemKey => {
    if (pathname.startsWith('/models')) return 'models'
    if (pathname.startsWith('/jobs')) return 'workspace'
    return 'home'
  }
  const active = activeKey()

  const handleClick = (key: ItemKey) => {
    if (key === 'workspace') {
      const id = currentJobId ?? jobs?.find(j => !j.archived)?.id
      if (id) {
        navigate({ to: '/jobs/$jobId', params: { jobId: id } })
        return
      }
      navigate({ to: '/' })
      return
    }
    navigate({ to: key === 'home' ? '/' : '/models' })
  }

  return (
    <nav
      aria-label="Primary"
      className="flex w-14 shrink-0 flex-col items-center gap-1 border-r bg-muted/30 py-3"
    >
      {!isMacOS && (
        <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-md bg-primary/10 font-semibold text-primary">
          S
        </div>
      )}

      {!isMacOS && (
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={handleMenuClick}
          className="h-9 w-9 text-foreground/70 hover:text-foreground"
          title="Menu"
          aria-label="Menu"
        >
          <Menu className="size-4" />
        </Button>
      )}

      {ITEMS.map(({ key, label, icon: Icon }) => {
        const isActive = active === key
        return (
          <button
            key={key}
            type="button"
            title={label}
            aria-label={label}
            aria-current={isActive ? 'page' : undefined}
            onClick={() => handleClick(key)}
            className={cn(
              'flex h-12 w-12 items-center justify-center rounded-md transition-colors',
              isActive
                ? 'bg-accent text-foreground'
                : 'text-muted-foreground hover:bg-accent/60 hover:text-foreground'
            )}
          >
            <Icon className="size-5" />
          </button>
        )
      })}
    </nav>
  )
}
