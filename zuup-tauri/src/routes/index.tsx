import { useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'

import { Header } from '@/components/layout/header'
import { useIsMobile } from '@/hooks/use-mobile'
import { Main } from '@/components/layout/main'
import { DownloadList } from '@/components/download-list'
import { AddDownloadModal } from '@/components/add-download-modal'

export const Route = createFileRoute('/')({
  component: App,
})

function App() {
  const isMobile = useIsMobile()
  const [showAddModal, setShowAddModal] = useState(false)

  return (
    <>
      <Header fixed>
        <h3 className={isMobile ? 'text-lg font-semibold' : 'text-2xl'}>
          Download Manager
        </h3>
      </Header>
      <Main>
        <DownloadList onAddDownload={() => setShowAddModal(true)} />
        <AddDownloadModal
          isOpen={showAddModal}
          onClose={() => setShowAddModal(false)}
          onDownloadAdded={() => {
            // The download list will refresh automatically
          }}
        />
      </Main>
    </>
  )
}
