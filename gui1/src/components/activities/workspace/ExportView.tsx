import {
  Download,
  FileSpreadsheet,
  FileArchive,
  Check,
  Loader2,
} from "lucide-react"
import { useState } from "react"
import {
  exportResultsCsv,
  exportJobBundle,
  type BundleExportOptions,
} from "../../../services/jobs"
import { save } from "@tauri-apps/plugin-dialog"
import { Button } from "../../ui/button"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "../../ui/dialog"

interface ExportViewProps {
  open: boolean
  jobId: string
  jobDetail: any
  onClose: () => void
}

export function ExportView({
  open,
  jobId,
  jobDetail,
  onClose,
}: ExportViewProps) {
  const [isExporting, setIsExporting] = useState(false)
  const [exportType, setExportType] = useState<"csv" | "bundle">("csv")
  const [bundleOptions, setBundleOptions] = useState<BundleExportOptions>({
    include_ad: true,
    include_weights: true,
    include_results: true,
    include_likes: true,
    criterion_ids: jobDetail?.criteria.map((c: any) => c.id) ?? [],
    top_n: undefined,
  })
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const handleCsvExport = async () => {
    try {
      setIsExporting(true)
      setError(null)
      setSuccess(null)
      const path = await save({
        defaultPath: `${jobDetail?.job?.title ?? "job"}-results.csv`,
        filters: [{ name: "CSV Files", extensions: ["csv"] }],
      })
      if (path) {
        await exportResultsCsv(jobId, path)
        setSuccess("Results exported successfully!")
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to export CSV")
    } finally {
      setIsExporting(false)
    }
  }

  const handleBundleExport = async () => {
    try {
      setIsExporting(true)
      setError(null)
      setSuccess(null)
      const path = await save({
        defaultPath: `${jobDetail?.job?.title ?? "job"}.triquijob`,
        filters: [{ name: "Job Bundle", extensions: ["triquijob"] }],
      })
      if (path) {
        await exportJobBundle(jobId, bundleOptions, path)
        setSuccess("Job bundle exported successfully!")
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to export bundle")
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Export Job</DialogTitle>
          <DialogDescription>
            Choose what to export and where to save it.
          </DialogDescription>
        </DialogHeader>

        <div className="mt-4 flex gap-2">
          <Button
            type="button"
            onClick={() => setExportType("csv")}
            variant={exportType === "csv" ? "default" : "outline"}
            className="flex-1"
            size="sm"
          >
            <div className="flex items-center gap-2">
              <FileSpreadsheet className="size-5" />
              <span className="font-medium">Results CSV</span>
            </div>
          </Button>
          <Button
            type="button"
            onClick={() => setExportType("bundle")}
            variant={exportType === "bundle" ? "default" : "outline"}
            className="flex-1"
            size="sm"
          >
            <div className="flex items-center gap-2">
              <FileArchive className="size-5" />
              <span className="font-medium">Job Bundle</span>
            </div>
          </Button>
        </div>

        {exportType === "bundle" && (
          <div className="mt-4 space-y-3 rounded border p-4">
            <h3 className="font-medium">Bundle Options</h3>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={bundleOptions.include_ad}
                onChange={(e) =>
                  setBundleOptions({
                    ...bundleOptions,
                    include_ad: e.target.checked,
                  })
                }
                className="rounded border"
              />
              <span className="text-sm">Include Job Ad</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={bundleOptions.include_weights}
                onChange={(e) =>
                  setBundleOptions({
                    ...bundleOptions,
                    include_weights: e.target.checked,
                  })
                }
                className="rounded border"
              />
              <span className="text-sm">Include Weights</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={bundleOptions.include_results}
                onChange={(e) =>
                  setBundleOptions({
                    ...bundleOptions,
                    include_results: e.target.checked,
                  })
                }
                className="rounded border"
              />
              <span className="text-sm">Include Results</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={bundleOptions.include_likes}
                onChange={(e) =>
                  setBundleOptions({
                    ...bundleOptions,
                    include_likes: e.target.checked,
                  })
                }
                className="rounded border"
              />
              <span className="text-sm">Include Likes</span>
            </label>
          </div>
        )}

        {error && (
          <div className="mt-3 rounded border border-destructive bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {success && (
          <div className="mt-3 flex items-center gap-2 rounded border border-green-500 bg-green-500/10 p-3 text-sm text-green-500">
            <Check className="size-4" />
            {success}
          </div>
        )}

        <DialogFooter className="mt-4">
          <Button type="button" onClick={onClose} variant="outline" size="sm">
            Cancel
          </Button>
          <Button
            type="button"
            onClick={
              exportType === "csv" ? handleCsvExport : handleBundleExport
            }
            disabled={isExporting}
            variant="default"
            size="sm"
          >
            {isExporting ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Download className="size-4" />
            )}
            {exportType === "csv" ? "Export CSV" : "Export Bundle"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
