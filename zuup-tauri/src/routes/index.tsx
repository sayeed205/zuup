import { createFileRoute } from '@tanstack/react-router'
import { Header } from '@/components/layout/header'
import { useIsMobile } from '@/hooks/use-mobile'
import { Main } from '@/components/layout/main'

export const Route = createFileRoute('/')({
  component: App,
})

function App() {
  const isMobile = useIsMobile()

  return (
    <>
      <Header fixed>
        <h3 className={isMobile ? 'text-lg font-semibold' : 'text-2xl'}>
          Dashboard
        </h3>
      </Header>
      <Main>Contents here</Main>
    </>
  )
}
