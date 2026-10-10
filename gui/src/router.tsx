import {
  createHashHistory,
  createRootRoute,
  createRoute,
  createRouter,
} from '@tanstack/react-router'

import { MainWindow } from '@/components/layout/MainWindow'
import { JobsList } from '@/components/screens/JobsList'
import { JobView } from '@/components/screens/JobView'
import { ModelsView } from '@/components/screens/ModelsView'

const rootRoute = createRootRoute({ component: MainWindow })

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: JobsList,
})

const jobRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/jobs/$jobId',
  component: JobView,
})

const modelsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/models',
  component: ModelsView,
})

const routeTree = rootRoute.addChildren([indexRoute, jobRoute, modelsRoute])

// Hash history keeps reload/deep-link behaviour robust inside the Tauri webview.
export const router = createRouter({
  routeTree,
  history: createHashHistory(),
})

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
