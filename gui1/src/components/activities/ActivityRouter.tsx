import { useHashRouter } from "../../lib/hash-router"
import { ModelsActivity } from "./models/ModelsActivity"
import { HomeActivity } from "./HomeActivity"
import { WorkspaceActivity } from "./WorkspaceActivity"

export function ActivityRouter() {
  const { path } = useHashRouter()

  // Current routes: /jobs (Jobs page), /models (Models page), /job/:id
  // (Job Details page). The legacy /home and /workspace routes stay routable so
  // old deep links and saved sessions keep working.
  const jobMatch = path.match(/^\/(?:job|workspace)\/(.+)$/)
  const jobId = jobMatch ? jobMatch[1] : null

  if (jobId) {
    return <WorkspaceActivity jobId={jobId} />
  }
  if (path.startsWith("/models")) {
    return <ModelsActivity />
  }
  return <HomeActivity />
}
