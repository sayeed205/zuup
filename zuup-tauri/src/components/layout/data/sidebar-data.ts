import { LayoutDashboard } from 'lucide-react'

import type { NavGroup } from '@/components/layout/types'

export const sidebarData: Array<NavGroup> = [
  {
    title: 'General',
    items: [
      {
        title: 'Dashboard',
        icon: LayoutDashboard,
        url: '/',
      },
    ],
  },
]
