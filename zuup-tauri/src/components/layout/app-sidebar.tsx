import * as React from 'react'
import { platform } from '@tauri-apps/plugin-os'

import { ThemeToggle } from '../theme-toggle'
import { WindowControls } from './controls/window-controlls'
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarRail,
} from '@/components/ui/sidebar'
import { NavGroup } from '@/components/layout/nav-group'
import { sidebarData } from '@/components/layout/data/sidebar-data'

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const currentPlatform = platform()
  return (
    <Sidebar collapsible="icon" variant="sidebar" {...props}>
      {currentPlatform === 'macos' && <WindowControls platform="macos" />}
      {/* <SidebarHeader>

      </SidebarHeader> */}
      <SidebarContent>
        {sidebarData.map((p) => (
          <NavGroup key={p.title} {...p} />
        ))}
      </SidebarContent>
      <SidebarFooter>
        <ThemeToggle />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
