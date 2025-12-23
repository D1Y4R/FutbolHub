# Football Prediction Hub - Design Guidelines

## Design Approach
**System**: Material Design + Sports Platform Hybrid (ESPN/FotMob/SofaScore patterns)
**Rationale**: Data-dense sports application requires robust component system with emphasis on readability, quick scanning, and progressive disclosure through modals.

## Typography
- **Primary Font**: Inter via Google Fonts CDN
- **Hierarchy**: 
  - H1: text-4xl font-bold (Main headings)
  - H2: text-2xl font-semibold (Section titles)
  - H3: text-xl font-medium (Card headers)
  - Body: text-base (General content)
  - Stats/Numbers: text-lg font-bold tabular-nums (Match scores, predictions)
  - Labels: text-sm text-gray-400 (Metadata, timestamps)

## Layout System
**Spacing Units**: Tailwind 2, 4, 6, 8, 12, 16
- Card padding: p-6
- Section gaps: gap-6 to gap-8
- Container max-width: max-w-7xl
- Grid gaps: gap-4 for tight data, gap-6 for features

## Component Library

### Navigation
Top navigation bar with logo left, main nav center (Maçlar, Tahminler, İstatistikler, Ligler), dark theme toggle right. Sticky positioning.

### Fixture Cards
Horizontal card layout: Team logo | Team name | Score/Time | Team name | Team logo. Include match status badge, league indicator, prediction confidence percentage bar below. Use grid-cols-1 md:grid-cols-2 for listing.

### Statistics Modal/Popup
Full-screen overlay (md:max-w-4xl centered). Header with team names and logos. Tabbed interface (Genel, Form, H2H, Kadro). Data presented in comparison format: left column (home team) | metric | right column (away team). Use progress bars for comparative stats, mini sparklines for form trends.

### Prediction Cards
Larger cards featuring: Match info header, AI confidence meter (circular progress), key factors list with icons, recommended bet suggestion with odds display. Shadow elevation on hover.

### Data Visualization
- Bar charts for head-to-head comparisons
- Radial progress for win probability
- Mini line graphs for team form (last 5 matches)
- Color coding: Green (wins), Red (losses), Gray (draws)

### Icons
**Library**: Heroicons via CDN
Use outline style for navigation, solid for data indicators (shield for defense stats, target for attack stats, etc.)

## Dark Theme Specifications
- Background hierarchy: bg-gray-900 (main) → bg-gray-800 (cards) → bg-gray-700 (elevated elements)
- Text: text-white (primary), text-gray-300 (secondary), text-gray-500 (tertiary)
- Borders: border-gray-700
- Accent colors: Emerald for positive predictions, Red for negative, Amber for neutral

## Images

### Hero Section
**Image**: Dynamic stadium atmosphere photo - modern football stadium at night with floodlights, crowd atmosphere, slightly blurred for depth. Full-width, 60vh height.
**Overlay**: Dark gradient overlay (from transparent to bg-gray-900)
**Content on Hero**: Main headline "Futbol Tahmin Merkezi", subheading about AI-powered predictions, primary CTA button with backdrop-blur-md bg-white/10

### Team Logos
Small circular logos (w-8 h-8 for lists, w-16 h-16 for modals) with subtle border. Use placeholder comments for dynamic team logos.

### Background Patterns
Subtle football field line pattern overlay on main background (opacity-5) for sports context without distraction.

## Page Structure

**Main Dashboard**: Hero section → Live matches section (3-column grid) → Upcoming predictions (2-column cards) → Top leagues sidebar (sticky) → Statistics highlights section

**Modal Structure**: Backdrop blur overlay → Centered modal with slide-up animation → Close button top-right → Tab navigation → Scrollable content area → Action buttons footer

## Key Interactions
- Modal triggers from fixture card clicks
- Tab switching within statistics modal
- Expandable prediction reasoning sections
- Live score updates with pulse animation
- Skeleton loaders for data fetching

## Accessibility
- High contrast ratios for dark theme readability
- Focus states with outline-offset-2 
- ARIA labels for all interactive elements
- Keyboard navigation for modals (ESC to close)
- Turkish language attributes on HTML element