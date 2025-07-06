'use client';

import React, { ReactNode } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ThemeToggleButton } from './ThemeToggleButton';

// NavLink now accepts isExpanded to adjust its own style
const NavLink = ({ href, children, isExpanded }: { href: string; children: ReactNode; isExpanded: boolean; }) => {
    const pathname = usePathname();
    const isActive = pathname === href;

    const linkStyle: React.CSSProperties = {
        display: 'flex',
        alignItems: 'center',
        // When collapsed, center the content (the icon). Otherwise, align left.
        justifyContent: isExpanded ? 'flex-start' : 'center',
        padding: '10px 15px',
        margin: '5px 10px',
        borderRadius: '6px',
        textDecoration: 'none',
        color: 'var(--foreground)',
        backgroundColor: isActive ? 'rgba(128, 128, 128, 0.1)' : 'transparent',
        whiteSpace: 'nowrap',
    };

    return (
        <Link href={href} style={linkStyle}>
            {children}
        </Link>
    );
};

interface SidebarProps {
  isExpanded: boolean;
}

export const Sidebar = ({ isExpanded }: SidebarProps) => {
  const navItems = [
    { href: '/', label: 'Chat', icon: '🗨️' },
    { href: '/internal-dialog', label: 'Internal Dialog', icon: '🧘' },
    { href: '/status', label: 'Status', icon: '⚡' },
    { href: '/memory', label: 'Memory', icon: '🌊' },
  ];

  const sidebarStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    height: '100vh',
    width: isExpanded ? '250px' : '68px',
    position: 'fixed',
    top: 0,
    left: 0,
    backgroundColor: 'var(--nav-background)',
    borderRight: '1px solid var(--nav-border)',
    transition: 'width 0.3s ease-in-out',
    overflowX: 'hidden',
    zIndex: 10,
  };

  return (
    <nav style={sidebarStyle}>
      <div>
        <div style={{ padding: '20px 15px', textAlign: 'center', fontWeight: 'bold', whiteSpace: 'nowrap', fontSize: '1.2rem', height: '60px' }}>
          {isExpanded ? 'swapnavue' : 'd'}
        </div>
        <ul style={{ padding: 0, margin: 0, marginTop: '10px' }}>
          {navItems.map(item => (
            <li key={item.href} style={{ listStyle: 'none' }}>
              {/* Pass isExpanded to each NavLink */}
              <NavLink href={item.href} isExpanded={isExpanded}>
                <span style={{ 
                    marginRight: isExpanded ? '15px' : '0', // Remove margin when collapsed
                    fontSize: '1.5rem', 
                    minWidth: '38px', 
                    textAlign: 'center' 
                }}>
                    {item.icon}
                </span>
                {isExpanded && item.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </div>

      <div>
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '10px' }}>
          <ThemeToggleButton />
        </div>
      </div>
    </nav>
  );
};