import React from 'react'
import { NavLink } from 'react-router-dom'
import {
  Home,
  Play,
  Layers,
  CheckCircle,
  Database,
  Brain,
  Activity,
  Briefcase,
  ChevronLeft,
  ChevronRight,
  BookOpen,
  Settings,
  HelpCircle
} from 'lucide-react'

const navigation = [
  { name: 'Generator', href: '/generate', icon: Play },
  { name: 'Batch', href: '/batch', icon: Layers },
  { name: 'Verify', href: '/verify', icon: CheckCircle },
  { name: 'Datasets', href: '/datasets', icon: Database },
  { name: 'ML Pipeline', href: '/ml', icon: Brain },
  { name: 'Monitor', href: '/monitor', icon: Activity },
  { name: 'Jobs', href: '/jobs', icon: Briefcase },
]

const resources = [
  { name: 'Documentation', href: '/docs', icon: BookOpen },
  { name: 'Settings', href: '/settings', icon: Settings },
  { name: 'Help', href: '/help', icon: HelpCircle },
]

export default function Sidebar({ open }) {
  return (
    <div className={`fixed inset-y-0 left-0 z-30 flex flex-col bg-gray-900 transition-all duration-300 ${open ? 'w-64' : 'w-16'}`}>
      <div className="flex-1 flex flex-col pt-5 pb-4 overflow-y-auto">
        <nav className="mt-5 flex-1 px-2 space-y-1">
          {navigation.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              className={({ isActive }) => `
                group flex items-center px-2 py-2 text-sm font-medium rounded-md
                ${isActive ? 'bg-gray-800 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'}
              `}
            >
              <item.icon className={`${open ? 'mr-3' : 'mx-auto'} flex-shrink-0 h-6 w-6`} />
              {open && <span>{item.name}</span>}
            </NavLink>
          ))}
        </nav>

        <div className="mt-auto">
          <div className="px-2 space-y-1">
            {resources.map((item) => (
              <NavLink
                key={item.name}
                to={item.href}
                className={({ isActive }) => `
                  group flex items-center px-2 py-2 text-sm font-medium rounded-md
                  ${isActive ? 'bg-gray-800 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'}
                `}
              >
                <item.icon className={`${open ? 'mr-3' : 'mx-auto'} flex-shrink-0 h-6 w-6`} />
                {open && <span>{item.name}</span>}
              </NavLink>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
