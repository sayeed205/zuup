import { useEffect, useState } from 'react'
import { platform as getPlatform } from '@tauri-apps/plugin-os'

import { TauriAppWindowProvider } from './contexts/plugin-window'

import { Windows } from './windows'
import { MacOS } from './mac'
import { Gnome } from './linux'

import type { WindowControlsProps } from './types'

import { cn } from '@/lib/utils'

export function WindowControls({
  platform,
  justify = false,
  hide = false,
  hideMethod = 'display',
  // linuxDesktop = "gnome",
  className,
  ...props
}: WindowControlsProps) {
  const [osType, setOsType] = useState<string | undefined>(undefined)

  useEffect(() => {
    const currentPlatform = getPlatform()
    setOsType(currentPlatform)
  }, [])

  const customClass = cn(
    'flex',
    className,
    hide && (hideMethod === 'display' ? 'hidden' : 'invisible'),
  )

  // Determine the default platform based on the operating system if not specified
  if (!platform) {
    switch (osType) {
      case 'macos':
        platform = 'macos'
        break
      case 'linux':
        platform = 'gnome'
        break
      default:
        platform = 'windows'
    }
  }

  const ControlsComponent = () => {
    switch (platform) {
      case 'windows':
        return (
          <Windows
            className={cn(customClass, justify && 'ml-auto')}
            {...props}
          />
        )
      case 'macos':
        return (
          <MacOS className={cn(customClass, justify && 'ml-0')} {...props} />
        )
      case 'gnome':
        return (
          <Gnome className={cn(customClass, justify && 'ml-auto')} {...props} />
        )
      default:
        return (
          <Windows
            className={cn(customClass, justify && 'ml-auto')}
            {...props}
          />
        )
    }
  }

  return (
    <TauriAppWindowProvider>
      <ControlsComponent />
    </TauriAppWindowProvider>
  )
}
