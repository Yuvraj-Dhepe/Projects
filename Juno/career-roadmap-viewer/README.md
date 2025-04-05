# Career Roadmap Explorer

A simple, visually appealing web application for students to explore career roadmaps using Mermaid diagrams. This application is built with Nuxt.js and optimized for deployment on Vercel.

## Features

- Interactive career roadmap diagrams
- Multiple viewing modes (PNG, SVG, Interactive)
- Responsive design for all devices
- Dark/Light theme support
- Fast loading and optimized performance
- Client-side rendering of Mermaid diagrams

## Setup

Make sure to install dependencies:

```bash
# npm
npm install

# pnpm
pnpm install

# yarn
yarn install

# bun
bun install
```

## Development Server

Start the development server on `http://localhost:3000`:

```bash
# npm
npm run dev

# pnpm
pnpm dev

# yarn
yarn dev

# bun
bun run dev
```

## Production

Build the application for production:

```bash
# npm
npm run build

# pnpm
pnpm build

# yarn
yarn build
```

Locally preview production build:

```bash
# npm
npm run preview

# pnpm
pnpm preview

# yarn
yarn preview
```

## Deployment to Vercel

This application is optimized for deployment on Vercel. To deploy:

1. Push your code to a GitHub repository
2. Connect your repository to Vercel
3. Vercel will automatically detect the Nuxt.js project and use the correct settings
4. The application will be built and deployed automatically

### Manual Deployment

You can also deploy manually using the Vercel CLI:

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy to production
vercel --prod
```

## Project Structure

- `components/` - Vue components including the MermaidViewer
- `layouts/` - Page layouts
- `pages/` - Application pages
- `public/` - Static assets
- `server/api/` - API endpoints for serving diagrams and career path data

## Adding New Career Paths

To add a new career path:

1. Create a new folder in the `career_roadmaps` directory with the career path name
2. Add a `diagrams` subfolder
3. Create a `career_path.mmd` file with the Mermaid diagram code
4. Add a `README.md` file with information about the career path
5. The application will automatically detect and display the new career path
