import { MessageSquare } from 'lucide-react'

function UserChatPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
      <div className="bg-white rounded-lg shadow-lg p-12 max-w-md">
        <MessageSquare className="w-16 h-16 text-primary-500 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          User Chat
        </h2>
        <p className="text-gray-600">
          Chat functionality coming soon...
        </p>
      </div>
    </div>
  )
}

export default UserChatPage

