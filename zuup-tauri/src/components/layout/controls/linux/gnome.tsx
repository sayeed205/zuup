import { useContext } from 'react'

import TauriAppWindowContext from '../contexts/plugin-window'

import type { HTMLProps } from 'react'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Icons } from '@/components/icons'

export function Gnome({ className, ...props }: HTMLProps<HTMLDivElement>) {
  const { isWindowMaximized, minimizeWindow, maximizeWindow, closeWindow } =
    useContext(TauriAppWindowContext)

  return (
    <div
      className={cn('mr-[10px] h-auto items-center space-x-[13px]', className)}
      {...props}
    >
      <Button
        onClick={minimizeWindow}
        variant="ghost"
        className="m-0 aspect-square h-6 w-6 cursor-default rounded-full"
      >
        <Icons.minimizeWin className="h-[9px] w-[9px]" />
      </Button>
      <Button
        onClick={maximizeWindow}
        variant="ghost"
        className="m-0 aspect-square h-6 w-6 cursor-default rounded-full"
      >
        {!isWindowMaximized ? (
          <Icons.maximizeWin className="h-2 w-2" />
        ) : (
          <Icons.maximizeRestoreWin className="h-[9px] w-[9px]" />
        )}
      </Button>
      <Button
        onClick={closeWindow}
        variant="ghost"
        className="m-0 aspect-square h-6 w-6 cursor-default rounded-full"
      >
        <Icons.closeWin className="h-2 w-2" />
      </Button>
    </div>
  )
}
