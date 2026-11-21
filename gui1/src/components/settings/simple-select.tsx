import * as React from "react"
import { cn } from "cn"
import { ChevronDownIcon } from "lucide-react"

interface SimpleSelectProps {
  value: string
  onValueChange: (value: string) => void
  children: React.ReactNode
  className?: string
  "data-testid"?: string
  "aria-label"?: string
}

export function SimpleSelect({
  value,
  onValueChange,
  children,
  className,
  "data-testid": testId,
  "aria-label": ariaLabel,
}: SimpleSelectProps) {
  return (
    <div className={cn("w-full", className)}>
      <select
        value={value}
        onChange={(e) => onValueChange(e.target.value)}
        className={cn(
          "flex w-full appearance-none items-center justify-between gap-1.5 rounded-lg border border-input bg-transparent py-1.5 pr-8 pl-2.5 text-sm leading-none whitespace-nowrap transition-colors outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-input/30 dark:hover:bg-input/50",
          "h-9"
        )}
        data-testid={testId}
        aria-label={ariaLabel}
      >
        {React.Children.map(children, (child) => {
          if (
            React.isValidElement(child) &&
            child.type === SimpleSelectOption
          ) {
            return child
          }
          return child
        })}
      </select>
      <ChevronDownIcon className="pointer-events-none absolute top-1/2 right-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
    </div>
  )
}

interface SimpleSelectOptionProps {
  value: string
  children: React.ReactNode
  "data-testid"?: string
}

export function SimpleSelectOption({
  value,
  children,
  "data-testid": testId,
}: SimpleSelectOptionProps) {
  return (
    <option value={value} data-testid={testId}>
      {children}
    </option>
  )
}

SimpleSelect.Option = SimpleSelectOption
