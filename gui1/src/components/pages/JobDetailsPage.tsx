import { WorkspaceActivity } from "@/components/activities/WorkspaceActivity"

export function JobDetailsPage({ jobId }: { jobId: string | null }) {
  return <WorkspaceActivity jobId={jobId} />
}
