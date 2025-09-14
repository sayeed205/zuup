import { CloudDownloadIcon, HouseIcon } from 'lucide-react'

import type { NavGroup } from '@/components/layout/types'

export const sidebarData: Array<NavGroup> = [
  {
    items: [
      {
        title: 'Home',
        icon: HouseIcon,
        url: '/',
      },
      {
        title: 'Downloads',
        icon: CloudDownloadIcon,
        url: '/downloads',
      },
    ],
  },
]
