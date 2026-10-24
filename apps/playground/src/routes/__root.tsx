import { createRootRoute, Link, Outlet } from "@tanstack/react-router";

import { MODEL_PAGES } from "@/models/registry";

const RootLayout = () => (
  <div className="min-h-screen bg-background text-foreground">
    <header className="border-b">
      <nav className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3">
        <Link to="/" className="font-semibold">
          Karume playground
        </Link>
        {MODEL_PAGES.map((page) => (
          <Link
            key={page.path}
            to={page.path}
            className="text-sm text-muted-foreground [&.active]:text-foreground"
          >
            {page.title}
          </Link>
        ))}
      </nav>
    </header>
    <main className="mx-auto max-w-7xl px-4 py-6">
      <Outlet />
    </main>
  </div>
);

export const Route = createRootRoute({ component: RootLayout });
