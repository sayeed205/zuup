import { createFileRoute } from '@tanstack/react-router'

import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'

export const Route = createFileRoute('/downloads')({
  component: RouteComponent,
})

function RouteComponent() {
  return (
    <>
      <Header fixed>
        <h3>Zuup Download Manager</h3>
      </Header>
      <Main>list of downloads</Main>
    </>
  )
}
