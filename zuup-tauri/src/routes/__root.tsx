import { Outlet, createRootRouteWithContext } from '@tanstack/react-router'
import { TanStackRouterDevtoolsPanel } from '@tanstack/react-router-devtools'
import Cookies from 'js-cookie'
import { TanstackDevtools } from '@tanstack/react-devtools'
import { platform } from '@tauri-apps/plugin-os'

import TanStackQueryDevtools from '../integrations/tanstack-query/devtools'

import type { QueryClient } from '@tanstack/react-query'
import { Toaster } from '@/components/ui/sonner'
import { AppSidebar } from '@/components/layout/app-sidebar'
import { cn } from '@/lib/utils'
import { SidebarProvider } from '@/components/ui/sidebar'
import { WindowControls } from '@/components/layout/controls/window-controlls'

interface MyRouterContext {
  queryClient: QueryClient
}

const defaultOpen = Cookies.get('sidebar_state') !== 'false'
const currentPlatform = platform()

export const Route = createRootRouteWithContext<MyRouterContext>()({
  component: () => (
    <>
      <div className="fixed right-1 top-1">
        {currentPlatform === 'linux' && <WindowControls platform="gnome" />}
        {currentPlatform === 'windows' && <WindowControls platform="windows" />}
      </div>
      <SidebarProvider defaultOpen={defaultOpen}>
        <AppSidebar />
        <div
          id="content"
          className={cn(
            'ml-auto w-full max-w-full',
            'peer-data-[state=collapsed]:w-[calc(100%-var(--sidebar-width-icon)-1rem)]',
            'peer-data-[state=expanded]:w-[calc(100%-var(--sidebar-width))]',
            'sm:transition-[width] sm:duration-200 sm:ease-linear',
            'flex h-svh flex-col',
            'group-data-[scroll-locked=1]/body:h-full',
            'has-[main.fixed-main]:group-data-[scroll-locked=1]/body:h-svh',
          )}
        >
          <Outlet />
        </div>
      </SidebarProvider>
      <Toaster duration={5000} />
      {import.meta.env.DEV && (
        <TanstackDevtools
          config={{
            position: 'bottom-right',
            hideUntilHover: true,
          }}
          plugins={[
            {
              name: 'Tanstack Router',
              render: <TanStackRouterDevtoolsPanel />,
            },
            TanStackQueryDevtools,
          ]}
        />
      )}
    </>
  ),
})
