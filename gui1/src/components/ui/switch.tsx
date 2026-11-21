import type { ComponentPropsWithoutRef } from "react"
import { Switch, SwitchThumb } from "@radix-ui/react-switch"

type SwitchComponentProps = {
  checked: boolean
  onChange: (checked: boolean) => void
  disabled?: boolean
} & Omit<
  ComponentPropsWithoutRef<typeof Switch>,
  "checked" | "onCheckedChange" | "onChange" | "disabled"
>

const SwitchComponent = ({
  checked,
  onChange,
  disabled,
  ...props
}: SwitchComponentProps) => (
  <Switch
    checked={checked}
    onCheckedChange={onChange}
    disabled={disabled}
    className="focus-visible:ring-primary-500 relative inline-flex h-[18px] w-[30px] shrink-0 cursor-pointer items-center rounded-full bg-gray-300 transition-colors focus-visible:ring-2 focus-visible:outline-none data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50 data-[state=checked]:bg-primary"
    {...props}
  >
    <SwitchThumb className="pointer-events-none block size-3 translate-x-[3px] rounded-full bg-white shadow transition-transform data-[state=checked]:translate-x-[15px]" />
  </Switch>
)

export { SwitchComponent }
