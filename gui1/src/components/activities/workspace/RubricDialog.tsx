import { useState } from "react"
import type { Job } from "../../../types"
import { Button } from "../../ui/button"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "../../ui/dialog"
import { Input } from "../../ui/input"
import { Label } from "../../ui/label"

interface RubricDialogProps {
  open: boolean
  job: Job | null
  onClose: () => void
  onUpdateWeights: (weights: {
    adWeight: number
    fbJobWeight: number
    fbProfileWeight: number
  }) => void
}

export function RubricDialog({
  open,
  job,
  onClose,
  onUpdateWeights,
}: RubricDialogProps) {
  const [adWeight, setAdWeight] = useState(job?.ad_weight ?? 50)
  const [fbJobWeight, setFbJobWeight] = useState(job?.fb_job_weight ?? 10)
  const [fbProfileWeight, setFbProfileWeight] = useState(
    job?.fb_profile_weight ?? 10
  )

  const handleSave = () => {
    onUpdateWeights({ adWeight, fbJobWeight, fbProfileWeight })
    onClose()
  }

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Rubric Weights</DialogTitle>
          <DialogDescription>
            Adjust max points for each rubric question.
          </DialogDescription>
        </DialogHeader>

        <div className="mt-4 space-y-4">
          <div>
            <Label>Job Ad Weight</Label>
            <div className="flex items-center gap-2">
              <Input
                type="number"
                value={adWeight}
                onChange={(e) => setAdWeight(parseFloat(e.target.value) || 0)}
                min="0"
                step="1"
                className="w-full"
              />
              <span className="text-xs text-muted-foreground">max points</span>
            </div>
          </div>

          <div>
            <Label>Job Fit Weight</Label>
            <div className="flex items-center gap-2">
              <Input
                type="number"
                value={fbJobWeight}
                onChange={(e) =>
                  setFbJobWeight(parseFloat(e.target.value) || 0)
                }
                min="0"
                step="1"
                className="w-full"
              />
              <span className="text-xs text-muted-foreground">max points</span>
            </div>
          </div>

          <div>
            <Label>Cross-Job Feedback Weight</Label>
            <div className="flex items-center gap-2">
              <Input
                type="number"
                value={fbProfileWeight}
                onChange={(e) =>
                  setFbProfileWeight(parseFloat(e.target.value) || 0)
                }
                min="0"
                step="1"
                className="w-full"
              />
              <span className="text-xs text-muted-foreground">max points</span>
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button type="button" onClick={onClose} variant="outline" size="sm">
            Cancel
          </Button>
          <Button
            type="button"
            onClick={handleSave}
            variant="default"
            size="sm"
          >
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
